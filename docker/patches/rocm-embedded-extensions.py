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
    # Upstream sources use both the angle-bracket and quoted include forms for
    # this header (e.g. cuda_utils/map_process.h and the vendored
    # diff-gaussian-rasterizer-depth/simple-knn sources use the quoted form,
    # which the angle-bracket-only match below used to silently miss --
    # letting `#include "device_launch_parameters.h"` survive hipification
    # unguarded and fail every ROCm build with "file not found").
    for device_header in (
        "#include <device_launch_parameters.h>",
        '#include "device_launch_parameters.h"',
    ):
        if device_header in text:
            text = text.replace(
                device_header,
                "#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n"
                f"{device_header}\n"
                "#endif",
            )

    # Some upstream CUDA sources use visually spaced launch chevrons. hipify's
    # parser expects normal CUDA launch syntax.
    text = text.replace("<< <", "<<<").replace(">> >", ">>>")

    # __trap() is a CUDA-only device intrinsic (used once in the vendored
    # diff-gaussian-rasterizer-depth/cuda_rasterizer/auxiliary.h to abort a
    # broken kernel invocation); HIP has no __trap(). __builtin_trap() is the
    # portable equivalent supported by both nvcc and hipcc/clang, so this
    # substitution needs no backend guard (confirmed identical bug and fix in
    # SplaTAM's own vendored copy of the same rasterizer source).
    text = text.replace("__trap();", "__builtin_trap();")

    # cuda_utils/map_process.cu (RTG's own pinned cuda_utils submodule, not
    # the rasterizer) defines its own device-side float atomicMax/atomicMin
    # overloads because upstream CUDA's <cuda_runtime.h> doesn't provide
    # float atomics. ROCm 7.2's own hip/amd_detail/amd_hip_atomic.h now
    # ships native `atomicMax(float*, float)` / `atomicMin(float*, float)`
    # overloads, so hipifying this file unchanged causes a hard
    # "redefinition of 'atomicMax'/'atomicMin'" compile error against the
    # platform header. Guard RTG's own definitions to CUDA-only; HIP builds
    # fall through to the native ROCm overloads instead (verified identical
    # semantics: both are a float-reinterpret-as-int atomicCAS spin loop).
    for name, cmp_op in (("atomicMax", ">"), ("atomicMin", "<")):
        needle = (
            f"// float {name}\n"
            f"__device__ __forceinline__ float {name}(float *address, float val)\n"
            "{\n"
            "    int ret = __float_as_int(*address);\n"
            f"    while (val {cmp_op} __int_as_float(ret))\n"
            "    {\n"
            "        int old = ret;\n"
            "        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)\n"
            "            break;\n"
            "    }\n"
            "    return __int_as_float(ret);\n"
            "}\n"
        )
        if needle in text:
            guarded = (
                "#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)\n"
                f"{needle}"
                "#endif\n"
            )
            text = text.replace(needle, guarded)

    # diff-gaussian-rasterizer-depth's forward.cu/backward.cu build a launch
    # grid with brace-init: `dim3 grid{tile_num, 1, 1};` where tile_num is a
    # plain `int`. nvcc silently narrows this to dim3's uint32_t fields;
    # hipcc's clang++ frontend enforces C++11 list-initialization narrowing
    # rules strictly and treats it as a hard error ("non-constant-expression
    # cannot be narrowed from type 'int' to 'uint32_t'"). Switching to
    # parenthesized construction sidesteps list-initialization entirely
    # (identical resulting value; dim3's constructor already takes
    # unsigned int and performs the same implicit conversion) and compiles
    # unchanged under nvcc too.
    text = text.replace(
        "dim3 grid{tile_num, 1, 1};",
        "dim3 grid(tile_num, 1, 1);",
    )

    # simple-knn's simple_knn.cu uses FLT_MAX without including <cfloat>
    # itself, relying on it arriving transitively through
    # #include "cuda_runtime.h". Under CUDA that pulls it in; under HIP,
    # hipify_python rewrites that include to hip/hip_runtime.h, whose
    # include chain does not transitively define FLT_MAX, so the same
    # source fails with "use of undeclared identifier 'FLT_MAX'" only on
    # ROCm. <cfloat> is portable (both nvcc and hipcc/clang) and needs no
    # backend guard.
    if "FLT_MAX" in text and "#include <cfloat>" not in text and "#include <float.h>" not in text:
        for anchor in ('#include "cuda_runtime.h"', "#include <cuda_runtime.h>"):
            if anchor in text:
                text = text.replace(anchor, f"{anchor}\n#include <cfloat>", 1)
                break

    if text != original:
        path.write_text(text)


for path in Path(".").rglob("*"):
    if path.suffix in {".cu", ".cuh", ".h", ".hpp"} and path.is_file():
        patch_file(path)
