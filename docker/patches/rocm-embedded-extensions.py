#!/usr/bin/env python3
"""Minimal HIP portability edits for RTG-SLAM's pinned embedded CUDA extensions.

The RTG cuda-utils submodule is pinned by the parent repository at
24dcf0aec00f35229f247ac206a5c0b141e15991. PyTorch CUDAExtension performs the
CUDA-to-HIP translation; this script only removes CUDA-only convenience headers,
routes cooperative-groups includes to HIP, and normalizes launch syntax that
hipify cannot parse. It intentionally leaves algorithmic code unchanged.
"""
from __future__ import annotations

from pathlib import Path


def patch_file(path: Path) -> None:
    text = path.read_text()
    original = text

    cg = "#include <cooperative_groups.h>"
    if cg in text:
        text = text.replace(
            cg,
            "#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)\n"
            "#include <hip/hip_cooperative_groups.h>\n"
            "#else\n"
            "#include <cooperative_groups.h>\n"
            "#endif",
        )
    reduce_header = "#include <cooperative_groups/reduce.h>"
    if reduce_header in text:
        text = text.replace(
            reduce_header,
            "#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n"
            "#include <cooperative_groups/reduce.h>\n"
            "#endif",
        )
    device_header = "#include <device_launch_parameters.h>"
    if device_header in text:
        text = text.replace(
            device_header,
            "#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n"
            "#include <device_launch_parameters.h>\n"
            "#endif",
        )

    # Some upstream CUDA sources use visually spaced launch chevrons. hipify's
    # parser expects normal CUDA launch syntax.
    text = text.replace("<< <", "<<<").replace(">> >", ">>>")
    if text != original:
        path.write_text(text)


for path in Path(".").rglob("*"):
    if path.suffix in {".cu", ".cuh", ".h", ".hpp"} and path.is_file():
        patch_file(path)
