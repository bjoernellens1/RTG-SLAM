#!/usr/bin/env python3
"""GPU smoke for RTG-SLAM's three mapping CUDA extensions on ROCm."""
from __future__ import annotations

import json
import os

import torch
from simple_knn._C import distCUDA2
from cuda_utils import _C as rtg_cuda
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer

assert os.environ.get("EXPECT_BACKEND") == "rocm"
assert torch.cuda.is_available() and torch.version.hip, "ROCm torch GPU backend unavailable"
device = torch.device("cuda:0")

# exact simple-knn algorithm path
pts = torch.tensor(
    [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
    dtype=torch.float32,
    device=device,
)
d2 = distCUDA2(pts)
assert d2.is_cuda and torch.isfinite(d2).all()

# RTG-specific map accumulation extension: execute a tiny valid map so an
# import-only image cannot be mistaken for a working GPU port.
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

# RTG depth rasterizer forward/backward.
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
)
rasterizer = GaussianRasterizer(settings)
means3d = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
opacities = torch.tensor([[0.8]], device=device, requires_grad=True)
colors = torch.tensor([[0.25, 0.5, 0.75]], device=device, requires_grad=True)
scales = torch.tensor([[0.05, 0.05, 0.05]], device=device, requires_grad=True)
rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
outputs = rasterizer(
    means3D=means3d,
    opacities=opacities,
    colors_precomp=colors,
    scales=scales,
    rotations=rotations,
)
assert outputs and all(
    torch.isfinite(t).all()
    for t in outputs
    if isinstance(t, torch.Tensor) and t.is_floating_point()
)
(outputs[0].sum() + outputs[1].sum() * 1e-3).backward()
assert means3d.grad is not None and torch.isfinite(means3d.grad).all()

print(json.dumps({
    "backend": "rocm",
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "capabilities": [
        "rtg_cuda_utils_gpu",
        "rtg_depth_rasterizer_gpu",
        "simple_knn_gpu",
    ],
}, sort_keys=True))
