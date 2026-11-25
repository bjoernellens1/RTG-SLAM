#!/usr/bin/env python3
"""GPU smoke for RTG-SLAM's three mapping extensions on CUDA or ROCm."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from pytorch3d.ops import knn_points
from simple_knn._C import distCUDA2
from cuda_utils import _C as rtg_cuda
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer

expected = os.environ.get("EXPECT_BACKEND", "").strip().lower()
assert expected in {"rocm", "cuda"}, "EXPECT_BACKEND must be rocm or cuda"
assert torch.cuda.is_available(), "PyTorch GPU backend unavailable"
if expected == "rocm":
    assert torch.version.hip, "ROCm image must expose torch.version.hip"
else:
    assert torch.version.hip is None, "CUDA image unexpectedly reports a HIP runtime"
device = torch.device("cuda:0")

# Exact simple-knn algorithm path.
pts = torch.tensor(
    [[0.0, 0.0, 1.0], [0.13, 0.0, 1.0], [0.0, 0.21, 1.0], [0.32, 0.15, 1.0]],
    dtype=torch.float32,
    device=device,
)
# RTG's own simple-knn spatial.cu binds distCUDA2 as
# std::tuple<torch::Tensor, torch::Tensor> (mean_dist, knn_indices) -- unlike
# upstream 3DGS's distCUDA2, which returns a single tensor.
d2_mean, d2_indices = distCUDA2(pts)
assert d2_mean.is_cuda and torch.isfinite(d2_mean).all()
assert d2_indices.is_cuda

# The mapper filters temporary points with cross-cloud K=3 and checks isolated
# points with self-query K=6. Use dyadic coordinates for exact reference dists.
refs = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0],
      [7.0, 0.0, 0.0], [15.0, 0.0, 0.0], [31.0, 0.0, 0.0], [63.0, 0.0, 0.0]]],
    dtype=torch.float32,
    device=device,
)
queries = torch.tensor([[[0.25, 0.5, 0.0], [6.75, 0.0, -0.5]]], device=device)
mapper_knn_tensors = {}
for label, query, reference, neighbors in (
    ("cross", queries, refs, 3),
    ("self", refs, refs, 6),
    ("short", queries, refs[:, :2], 3),
):
    dists, indices, points = knn_points(query, reference, norm=2, K=neighbors, return_nn=True)
    assert dists.is_cuda and indices.is_cuda and points.is_cuda
    valid = min(neighbors, reference.shape[1])
    matrix = (query[:, :, None] - reference[:, None]).square().sum(-1)
    expected_dist, expected_index = matrix.topk(valid, dim=-1, largest=False)
    torch.testing.assert_close(dists[..., :valid], expected_dist, rtol=0, atol=0)
    torch.testing.assert_close(indices[..., :valid], expected_index, rtol=0, atol=0)
    torch.testing.assert_close(points[..., :valid, :], reference[0][expected_index], rtol=0, atol=0)
    assert (dists[..., valid:] == 0).all() and (indices[..., valid:] == 0).all()
    assert (points[..., valid:, :] == 0).all()
    mapper_knn_tensors.update({
        f"mapper_knn_{label}_squared_distances": dists,
        f"mapper_knn_{label}_indices": indices,
        f"mapper_knn_{label}_points": points,
    })

# Compare tied membership by distance and index. Native order is diagnostic.
tie_refs = torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
tie_dist, tie_idx, _ = knn_points(refs[:, :1], tie_refs, norm=2, K=3, return_nn=True)
torch.testing.assert_close(tie_dist, torch.tensor([[[1.0, 1.0, 4.0]]], device=device), rtol=0, atol=0)
tie_idx_sorted, by_index = tie_idx.sort(dim=-1)
by_distance = tie_dist.gather(-1, by_index).argsort(dim=-1, stable=True)
canonical_tie_idx = tie_idx_sorted.gather(-1, by_distance)
torch.testing.assert_close(canonical_tie_idx, torch.arange(3, device=device).reshape(1, 1, 3))
radii = torch.tensor([1.0, 2.0, 1.0], device=device)
tie_inside = (tie_dist.sqrt() < radii[tie_idx] * 0.6).any(dim=-1)
assert tie_inside.all()
mapper_knn_tensors.update(
    mapper_knn_tie_squared_distances=tie_dist,
    mapper_knn_tie_indices=canonical_tie_idx,
    mapper_knn_tie_inside=tie_inside,
)

# RTG-specific map accumulation extension.
H = W = 2
P = 4
color_error = torch.arange(1, P+1, device=device, dtype=torch.float32).reshape(H,W) * 0.03
depth_error = color_error * 0.7
normal_error = color_error * 0.2
color_index = torch.arange(P, device=device, dtype=torch.int32).reshape(H, W)
depth_index = color_index.clone()
out = rtg_cuda.accumulate_gaussian_error(
    H, W, P,
    color_error, depth_error, normal_error,
    color_index, depth_index,
    1.0, 1.0, 1.0, False,
)
assert len(out) == 4 and all(t.is_cuda for t in out)
assert all(torch.isfinite(t).all() for t in out if t.is_floating_point())

# Match SLAM/render.py exactly: cx/cy are mandatory and tile_mask is int32.
settings = GaussianRasterizationSettings(
    image_height=16,
    image_width=16,
    tanfovx=1.0,
    tanfovy=1.0,
    bg=torch.zeros(3, device=device),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4, device=device),
    projmatrix=torch.eye(4, device=device),
    sh_degree=0,
    campos=torch.zeros(3, device=device),
    opaque_threshold=0.0,
    normal_threshold=1.0,
    depth_threshold=1.0,
    prefiltered=False,
    debug=False,
    cx=7.5,
    cy=7.5,
    color_sigma=3.0,
    T_threshold=0.0001,
)
rasterizer = GaussianRasterizer(settings)
means3d = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
opacities = torch.tensor([[0.8]], device=device, requires_grad=True)
colors = torch.tensor([[0.25, 0.5, 0.75]], device=device, requires_grad=True)
scales = torch.tensor([[0.05, 0.05, 0.05]], device=device, requires_grad=True)
rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
tile_mask = torch.ones((1, 1), device=device, dtype=torch.int32)
outputs = rasterizer(
    means3D=means3d,
    opacities=opacities,
    colors_precomp=colors,
    scales=scales,
    rotations=rotations,
    tile_mask=tile_mask,
)
assert len(outputs) == 8
assert all(
    torch.isfinite(t).all()
    for t in outputs
    if isinstance(t, torch.Tensor) and t.is_floating_point()
)
(outputs[0].sum() + outputs[1].sum() * 1e-3).backward()
assert means3d.grad is not None and torch.isfinite(means3d.grad).all()

torch.cuda.synchronize()
# Record actual tensors for cross-vendor numerical checks, not just finite flags.
if os.environ.get("RTG_SMOKE_ARRAYS"):
    tensors = {"knn_mean": d2_mean, "knn_indices": d2_indices}
    tensors.update(mapper_knn_tensors)
    tensors.update({f"accum_{i}": t for i, t in enumerate(out)})
    tensors.update({f"render_{i}": t for i, t in enumerate(outputs) if isinstance(t, torch.Tensor)})
    for name, t in (("means", means3d), ("opacity", opacities), ("colors", colors),
                    ("scales", scales), ("rotation", rotations)):
        if t.grad is not None:
            tensors[f"grad_{name}"] = t.grad
    np.savez(os.environ["RTG_SMOKE_ARRAYS"], **{k: v.detach().cpu().numpy() for k,v in tensors.items()})

print(json.dumps({
    "backend": expected,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "mapper_knn_native_tie_indices": tie_idx.cpu().tolist(),
    "capabilities": [
        "rtg_cuda_utils_gpu",
        "rtg_depth_rasterizer_gpu",
        "simple_knn_gpu",
        "pytorch3d_mapper_knn_gpu",
    ],
}, sort_keys=True))
