from pathlib import Path


def test_mapper_guards_empty_render_reductions_and_clears_gradients():
    source = Path("SLAM/multiprocess/mapper.py").read_text()

    assert "color_pixels.numel() > 0" in source
    assert "valid_depth.numel() > 0" in source
    assert "valid_normal.numel() > 0" in source
    assert "self.optimizer.zero_grad(set_to_none=True)" in source
    assert "if update_loss.requires_grad:" in source
    assert "if pointcloud._features_dc.grad is not None:" in source


def test_mapper_skips_zero_tile_objectives_before_render():
    source = Path("SLAM/multiprocess/mapper.py").read_text()
    guard = "if not tile_mask_has_work[random_index]:"

    assert "tile_mask_has_work.append(" in source
    assert guard in source
    assert source.index(guard) < source.index("render_ouput = self.renderer.render")
