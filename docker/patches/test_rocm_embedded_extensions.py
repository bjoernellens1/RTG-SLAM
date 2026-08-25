"""Regression tests for the ROCm extension-source rewrite."""

from pathlib import Path
import subprocess
import sys


def test_empty_flat_raster_skips_zero_grid_and_marks_no_hit(tmp_path: Path) -> None:
    source_root = Path(__file__).parent
    rasterizer = tmp_path / "diff-gaussian-rasterizer-depth"
    rasterizer.mkdir()
    forward = rasterizer / "forward.cu"
    forward.write_text(
        "void render_flat(const int tile_num)\n{\n\tdim3 grid(tile_num, 1, 1);\n}\n"
    )
    points = rasterizer / "rasterize_points.cu"
    points.write_text(
        "torch::Tensor out_hit_depth = torch::full({1, H, W}, 0, int_opts);\n"
        "torch::Tensor out_hit_color = torch::full({1, H, W}, 0, int_opts);\n"
    )
    implementation = rasterizer / "rasterizer_impl.cu"
    implementation.write_text(
        "__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)\n"
        "{\n"
        "\tauto idx = cg::this_grid().thread_rank();\n"
        "\tuint32_t currtile = key >> 32;\n"
        "}\n"
        "int num_rendered;\n"
        "CHECK_CUDA(cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);\n"
        "identifyTileRanges<<<1, 1>>>(num_rendered, point_list_keys, ranges);\n"
        "uint2 ranges_cpu[tile_grid.x * tile_grid.y];\n"
        "CHECK_CUDA(cudaMemcpy(ranges_cpu, ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost), debug);\n"
        "std::vector<int> tile_indices_cpu;\n"
        "CHECK_CUDA(cudaMemcpy(tile_indices, tile_indices_cpu.data(), tile_indices_cpu.size() * sizeof(int), cudaMemcpyHostToDevice), debug);\n"
        "tile_num = tile_indices_cpu.size();\n"
        "void backward()\n{\n"
        "\tGeometryState geomState = GeometryState::fromChunk(geom_buffer, P);\n"
        "}\n"
    )
    auxiliary = rasterizer / "auxiliary.h"
    auxiliary.write_text(
        "#define CHECK_CUDA(A, debug) \\\n"
        "\tA; \\\n"
        "\tif (debug) {}\n"
    )

    subprocess.run([sys.executable, source_root / "rocm-embedded-extensions.py"], cwd=tmp_path, check=True)

    assert "if (tile_num == 0) return;" in forward.read_text()
    assert "out_hit_depth = torch::full({1, H, W}, -1, int_opts);" in points.read_text()
    assert "out_hit_color = torch::full({1, H, W}, -1, int_opts);" in points.read_text()
    assert "if (!tile_indices_cpu.empty())" in implementation.read_text()
    assert "if (tile_num == 0)\n\t\treturn num_rendered;" in implementation.read_text()
    assert "if (tile_num == 0) return;\n\tGeometryState geomState" in implementation.read_text()
    assert "uint2 *ranges, const uint32_t tile_count, const bool debug)" in implementation.read_text()
    assert "auto idx = blockIdx.x * blockDim.x + threadIdx.x;" in implementation.read_text()
    assert "if (currtile >= tile_count)" in implementation.read_text()
    assert '"[RTG_BINNING] P=" << P << " num_rendered=" << num_rendered' in implementation.read_text()
    assert "if (num_rendered == 0)\n\t{\n\t\ttile_num = 0;\n\t\treturn 0;\n\t}" in implementation.read_text()
    assert "static thread_local int *num_rendered_host = nullptr;" in implementation.read_text()
    assert "static thread_local uint2 *ranges_cpu = nullptr;" in implementation.read_text()
    assert "static thread_local size_t ranges_cpu_capacity = 0;" in implementation.read_text()
    assert "if (ranges_cpu_capacity < ranges_bytes)" in implementation.read_text()
    assert "cudaFreeHost" not in implementation.read_text()
    assert "point_list_keys, ranges, tile_grid.x * tile_grid.y, debug);" in implementation.read_text()
    assert '"[RTG_NATIVE_STAGE] " << __FILE__ << ":" << __LINE__' in auxiliary.read_text()


def test_scan_workspace_query_matches_out_of_place_execution(tmp_path: Path) -> None:
    source_root = Path(__file__).parent
    rasterizer = tmp_path / "diff-gaussian-rasterizer-depth"
    rasterizer.mkdir()
    implementation = rasterizer / "rasterizer_impl.cu"
    implementation.write_text(
        "\tobtain(chunk, geom.tiles_touched, P, 128);\n"
        "\tcub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);\n"
        "\tobtain(chunk, geom.scanning_space, geom.scan_size, 128);\n"
        "\tobtain(chunk, geom.point_offsets, P, 128);\n"
    )

    subprocess.run([sys.executable, source_root / "rocm-embedded-extensions.py"], cwd=tmp_path, check=True)

    assert implementation.read_text() == (
        "\tobtain(chunk, geom.tiles_touched, P, 128);\n"
        "\tobtain(chunk, geom.point_offsets, P, 128);\n"
        "\tcub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.point_offsets, P);\n"
        "\tobtain(chunk, geom.scanning_space, geom.scan_size, 128);\n"
    )
