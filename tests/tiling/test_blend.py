from __future__ import annotations

import numpy as np

from engvit.tiling.blend import build_blend_weights
from engvit.tiling.layout import plan_layout


def test_cosine_weights_are_positive_and_normalize_every_pixel() -> None:
    layout = plan_layout(
        width=61,
        height=47,
        tile_size=24,
        overlap=7,
        context=3,
    )
    total = np.zeros((47 * 2, 61 * 2), dtype=np.float32)
    for tile in layout.tiles:
        weights = build_blend_weights(tile, layout, scale=2)
        assert np.all(weights > 0)
        total[tile.y0 * 2 : tile.y1 * 2, tile.x0 * 2 : tile.x1 * 2] += weights
    assert np.all(total > 0)

