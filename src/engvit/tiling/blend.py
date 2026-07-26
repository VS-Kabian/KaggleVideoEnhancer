"""Strictly-positive separable cosine blend masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engvit.tiling.layout import TileLayout, TileRegion


def _ramp(length: int) -> NDArray[np.float32]:
    if length <= 0:
        return np.ones((0,), dtype=np.float32)
    position = (np.arange(length, dtype=np.float32) + np.float32(0.5)) / length
    return (np.float32(0.5) - np.float32(0.5) * np.cos(np.pi * position)).astype(
        np.float32
    )


def build_blend_weights(
    tile: TileRegion,
    layout: TileLayout,
    *,
    scale: int,
) -> NDArray[np.float32]:
    """Build a tile-core mask using its actual neighbor overlap widths."""
    if scale < 1:
        raise ValueError("blend scale must be positive")
    width = (tile.x1 - tile.x0) * scale
    height = (tile.y1 - tile.y0) * scale
    x_weight = np.ones(width, dtype=np.float32)
    y_weight = np.ones(height, dtype=np.float32)
    left = tile.overlap_left * scale
    right = tile.overlap_right * scale
    top = tile.overlap_top * scale
    bottom = tile.overlap_bottom * scale
    if left:
        x_weight[:left] *= _ramp(left)
    if right:
        x_weight[-right:] *= _ramp(right)[::-1]
    if top:
        y_weight[:top] *= _ramp(top)
    if bottom:
        y_weight[-bottom:] *= _ramp(bottom)[::-1]
    weights = y_weight[:, None] * x_weight[None, :]
    if weights.shape != (height, width) or np.any(weights <= 0):
        raise ValueError("blend mask contains invalid or zero weights")
    return weights

