#!/usr/bin/env python3
"""GPU smoke for RTG-SLAM's three mapping extensions on CUDA or ROCm."""
from __future__ import annotations

import json
import os

import torch
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
    [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
    dtype=torch.float32,
    device=device,
)
# RTG's own simple-knn spatial.cu binds distCUDA2 as
# std::tuple<torch::Tensor, torch::Tensor> (mean_dist, knn_indices) -- unlike
# upstream 3DGS's distCUDA2, which returns a single tensor.
d2_mean, d2_indices = distCUDA2(pts)
assert d2_mean.is_cuda and torch.isfinite(d2_mean).all()
assert d2_indices.is_cuda

# RTG-specific map accumulation extension.
H = W = 2
P = 4
color_error = torch.zeros((H, W), device=device, dtype=torch.float32)
depth_error = torch.zeros((H, W), device=device, dtype=torch.float32)
normal_error = torch.zeros((H, W), device=device, dtype=torch.float32)
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

print(json.dumps({
    "backend": expected,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "capabilities": [
        "rtg_cuda_utils_gpu",
        "rtg_depth_rasterizer_gpu",
        "simple_knn_gpu",
    ],
}, sort_keys=True))
