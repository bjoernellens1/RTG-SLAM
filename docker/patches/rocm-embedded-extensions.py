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

    # RTG uses a compact list of tiles that contain at least one projected
    # Gaussian. A camera can legitimately see none of the previous map, in
    # which case launching the flat rasterizer with grid.x == 0 is invalid on
    # HIP. The caller has already allocated the empty render outputs, so skip
    # only that zero-work launch. Mark its hit maps invalid so the mapper does
    # not mistake Gaussian zero for a rendered point.
    if path.name in {"forward.cu", "backward.cu"}:
        text = text.replace(
            "{\n\tdim3 grid(tile_num, 1, 1);",
            "{\n\tif (tile_num == 0) return;\n\tdim3 grid(tile_num, 1, 1);",
        )
    # HIP does not reliably accept a zero-byte cudaMemcpy whose host pointer
    # is ``std::vector::data()`` from an empty vector.  The compact tile list
    # is empty for a valid no-visible-Gaussian view, so avoid that copy before
    # the zero-tile launch guard above takes effect.
    if path.name == "rasterizer_impl.cu":
        text = text.replace(
            "\tint num_rendered;",
            "\tint num_rendered;\n\tstatic thread_local int *num_rendered_host = nullptr;",
        )
        text = text.replace(
            "int num_rendered;\nCHECK_CUDA(cudaMemcpy(&num_rendered,",
            "int num_rendered;\nstatic thread_local int *num_rendered_host = nullptr;\nCHECK_CUDA(cudaMemcpy(&num_rendered,",
        )
        for offsets in ("point_offsets", "geomState.point_offsets"):
            copy_line = (
                f"CHECK_CUDA(cudaMemcpy(&num_rendered, {offsets} + P - 1, sizeof(int), "
                "cudaMemcpyDeviceToHost), debug);"
            )
            text = text.replace(
                copy_line,
                "if (num_rendered_host == nullptr)\n"
                "\t\tCHECK_CUDA(cudaMallocHost((void **)&num_rendered_host, sizeof(int)), debug);\n"
                f"\tCHECK_CUDA(cudaMemcpy(num_rendered_host, {offsets} + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);\n"
                "\tnum_rendered = *num_rendered_host;\n"
                "\tif (num_rendered == 0)\n"
                "\t{\n"
                "\t\ttile_num = 0;\n"
                "\t\treturn 0;\n"
                "\t}\n"
                "\tif (debug) std::cerr << \"[RTG_BINNING] P=\" << P << \" num_rendered=\" << num_rendered "
                "<< \" tiles=\" << tile_grid.x * tile_grid.y << std::endl;",
            )
        text = text.replace(
            "__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)",
            "__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges, const uint32_t tile_count, const bool debug)",
        )
        text = text.replace(
            "__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges, const uint32_t tile_count, const bool debug)\n"
            "{\n\tauto idx = cg::this_grid().thread_rank();",
            "__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges, const uint32_t tile_count, const bool debug)\n"
            "{\n\tauto idx = blockIdx.x * blockDim.x + threadIdx.x;",
        )
        text = text.replace(
            "\tuint32_t currtile = key >> 32;",
            "\tuint32_t currtile = key >> 32;\n"
            "\tif (currtile >= tile_count)\n"
            "\t{\n"
            "\t\tif (debug) printf(\"[RTG_INVALID_TILE_KEY] idx=%u currtile=%u tile_count=%u\\n\", (unsigned)idx, currtile, tile_count);\n"
            "\t\treturn;\n"
            "\t}",
        )
        text = text.replace(
            "\t\tuint32_t prevtile = point_list_keys[idx - 1] >> 32;",
            "\t\tuint32_t prevtile = point_list_keys[idx - 1] >> 32;\n"
            "\t\tif (prevtile >= tile_count)\n"
            "\t\t{\n"
            "\t\t\tif (debug) printf(\"[RTG_INVALID_TILE_KEY] idx=%u prevtile=%u tile_count=%u\\n\", (unsigned)idx, prevtile, tile_count);\n"
            "\t\t\treturn;\n"
            "\t\t}",
        )
        text = text.replace(
            "identifyTileRanges<<<1, 1>>>(num_rendered, point_list_keys, ranges);",
            "identifyTileRanges<<<1, 1>>>(num_rendered, point_list_keys, ranges, tile_grid.x * tile_grid.y, debug);",
        )
        text = text.replace(
            "\t\t\tnum_rendered,\n\t\t\tbinningState.point_list_keys,\n\t\t\timgState.ranges);",
            "\t\t\tnum_rendered,\n\t\t\tbinningState.point_list_keys,\n\t\t\timgState.ranges,\n"
            "\t\t\ttile_grid.x * tile_grid.y,\n\t\t\tdebug);",
        )
        text = text.replace(
            "uint2 ranges_cpu[tile_grid.x * tile_grid.y];",
            "static thread_local uint2 *ranges_cpu = nullptr;\n"
            "\tstatic thread_local size_t ranges_cpu_capacity = 0;\n"
            "\tconst size_t ranges_bytes = tile_grid.x * tile_grid.y * sizeof(uint2);\n"
            "\tif (ranges_cpu_capacity < ranges_bytes)\n"
            "\t{\n"
            "\t\tuint2 *larger_ranges_cpu;\n"
            "\t\tCHECK_CUDA(cudaMallocHost((void **)&larger_ranges_cpu, ranges_bytes), debug);\n"
            "\t\tranges_cpu = larger_ranges_cpu;\n"
            "\t\tranges_cpu_capacity = ranges_bytes;\n"
            "\t}",
        )
        text = text.replace(
            "CHECK_CUDA(cudaMemcpy(tile_indices, tile_indices_cpu.data(), tile_indices_cpu.size() * sizeof(int), cudaMemcpyHostToDevice), debug);",
            "if (!tile_indices_cpu.empty())\n"
            "\t\tCHECK_CUDA(cudaMemcpy(tile_indices, tile_indices_cpu.data(), tile_indices_cpu.size() * sizeof(int), cudaMemcpyHostToDevice), debug);",
        )
        text = text.replace(
            "tile_num = tile_indices_cpu.size();",
            "tile_num = tile_indices_cpu.size();\n"
            "\tif (tile_num == 0)\n"
            "\t\treturn num_rendered;",
        )
        text = text.replace(
            "{\n\tGeometryState geomState = GeometryState::fromChunk(geom_buffer, P);",
            "{\n\tif (tile_num == 0) return;\n"
            "\tGeometryState geomState = GeometryState::fromChunk(geom_buffer, P);",
        )
    if path.name == "auxiliary.h" and "#define CHECK_CUDA(A, debug)" in text:
        lines = text.splitlines(keepends=True)
        macro_index = next(
            index for index, line in enumerate(lines)
            if line.startswith("#define CHECK_CUDA(A, debug)")
        )
        action_index = macro_index + 1
        if lines[action_index].lstrip().startswith("A;"):
            indent = lines[action_index][:-len(lines[action_index].lstrip())]
            lines.insert(
                action_index,
                f'{indent}if (debug) std::cerr << "[RTG_NATIVE_STAGE] " << __FILE__ << ":" << __LINE__ << std::endl; \\\n',
            )
            text = "".join(lines)
    if path.name == "rasterize_points.cu":
        text = text.replace(
            "torch::Tensor out_hit_depth = torch::full({1, H, W}, 0, int_opts);",
            "torch::Tensor out_hit_depth = torch::full({1, H, W}, -1, int_opts);",
        )
        text = text.replace(
            "torch::Tensor out_hit_color = torch::full({1, H, W}, 0, int_opts);",
            "torch::Tensor out_hit_color = torch::full({1, H, W}, -1, int_opts);",
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
