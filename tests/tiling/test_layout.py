from __future__ import annotations

import numpy as np

from engvit.tiling.layout import plan_layout


def test_odd_dimensions_have_exact_symmetric_coverage() -> None:
    layout = plan_layout(
        width=101,
        height=73,
        tile_size=32,
        overlap=8,
        context=4,
    )
    coverage = np.zeros((73, 101), dtype=np.int32)
    for tile in layout.tiles:
        coverage[tile.y0 : tile.y1, tile.x0 : tile.x1] += 1
        assert 0 <= tile.read_x0 <= tile.x0 < tile.x1 <= tile.read_x1 <= 101
        assert 0 <= tile.read_y0 <= tile.y0 < tile.y1 <= tile.read_y1 <= 73
    assert np.all(coverage >= 1)
    assert layout.width == 101
    assert layout.height == 73
    assert layout.x_positions[0] == 0
    assert layout.x_positions[-1] == 69


def test_invalid_overlap_is_rejected() -> None:
    try:
        plan_layout(width=64, height=64, tile_size=32, overlap=32, context=0)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlap equal to tile size must fail")

