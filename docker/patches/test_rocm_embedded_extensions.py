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
        "CHECK_CUDA(cudaMemcpy(tile_indices, tile_indices_cpu.data(), tile_indices_cpu.size() * sizeof(int), cudaMemcpyHostToDevice), debug);\n"
    )

    subprocess.run([sys.executable, source_root / "rocm-embedded-extensions.py"], cwd=tmp_path, check=True)

    assert "if (tile_num == 0) return;" in forward.read_text()
    assert "out_hit_depth = torch::full({1, H, W}, -1, int_opts);" in points.read_text()
    assert "out_hit_color = torch::full({1, H, W}, -1, int_opts);" in points.read_text()
    assert "if (!tile_indices_cpu.empty())" in implementation.read_text()
