from pathlib import Path


def test_renderer_has_opt_in_extension_boundary_syncs():
    source = Path("SLAM/render.py").read_text()

    assert 'os.environ.get("RTG_DEBUG_SYNC") == "1"' in source
    assert '"[RTG_DEBUG_SYNC] rasterizer-return"' in source
    assert '"[RTG_DEBUG_SYNC] normal-index"' in source
