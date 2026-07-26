"""Distributional seam evidence without an arbitrary pass threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engvit.tiling.blend import build_blend_weights
from engvit.tiling.layout import TileLayout


@dataclass(frozen=True)
class SeamEvidence:
    boundary_p50: float
    boundary_p95: float
    boundary_p99: float
    interior_p50: float
    interior_p95: float
    interior_p99: float
    zero_weight_pixels: int


def _percentiles(values: NDArray[np.float32]) -> tuple[float, float, float]:
    if values.size == 0:
        return (0.0, 0.0, 0.0)
    result = np.percentile(values, (50, 95, 99))
    return (float(result[0]), float(result[1]), float(result[2]))


def analyze_seams(
    tiled: NDArray[np.uint8],
    reference: NDArray[np.uint8] | None,
    layout: TileLayout,
    *,
    scale: int,
) -> SeamEvidence:
    """Compare boundary and interior error distributions and weight coverage."""
    expected_shape = (layout.height * scale, layout.width * scale, 3)
    if tiled.shape != expected_shape:
        raise ValueError("tiled image shape does not match layout and scale")
    boundary_mask = np.zeros(expected_shape[:2], dtype=bool)
    for position in layout.x_positions[1:]:
        x = position * scale
        boundary_mask[:, max(0, x - 1) : min(expected_shape[1], x + 1)] = True
    for position in layout.y_positions[1:]:
        y = position * scale
        boundary_mask[max(0, y - 1) : min(expected_shape[0], y + 1), :] = True

    if reference is not None:
        if reference.shape != tiled.shape:
            raise ValueError("seam reference shape does not match tiled output")
        error = np.mean(
            np.abs(tiled.astype(np.float32) - reference.astype(np.float32)),
            axis=2,
        )
    else:
        luma = np.mean(tiled.astype(np.float32), axis=2)
        error = np.maximum(
            np.abs(np.diff(luma, axis=1, prepend=luma[:, :1])),
            np.abs(np.diff(luma, axis=0, prepend=luma[:1, :])),
        )
    boundary = _percentiles(error[boundary_mask])
    interior = _percentiles(error[~boundary_mask])

    weights = np.zeros(expected_shape[:2], dtype=np.float32)
    for tile in layout.tiles:
        tile_weights = build_blend_weights(tile, layout, scale=scale)
        weights[
            tile.y0 * scale : tile.y1 * scale,
            tile.x0 * scale : tile.x1 * scale,
        ] += tile_weights
    return SeamEvidence(
        boundary_p50=boundary[0],
        boundary_p95=boundary[1],
        boundary_p99=boundary[2],
        interior_p50=interior[0],
        interior_p95=interior[1],
        interior_p99=interior[2],
        zero_weight_pixels=int(np.sum(weights <= 0)),
    )

