"""Sequential bounded-GPU tiled enhancement with CPU accumulation."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from engvit.tiling.blend import build_blend_weights
from engvit.tiling.layout import plan_layout
from engvit.types import FrameEnhancer, TilePolicy


def enhance_tiled(
    frame: NDArray[np.uint8],
    enhancer: FrameEnhancer,
    policy: TilePolicy,
) -> NDArray[np.uint8]:
    """Enhance context-padded tiles and normalize overlapping output cores."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("tiled enhancer input must be HxWx3 RGB uint8")
    height, width = frame.shape[:2]
    scale = enhancer.scale
    layout = plan_layout(
        width=width,
        height=height,
        tile_size=policy.tile_size,
        overlap=policy.blend_overlap,
        context=policy.context_pad,
    )
    accumulation = np.zeros(
        (height * scale, width * scale, 3),
        dtype=np.float32,
    )
    total_weight = np.zeros((height * scale, width * scale), dtype=np.float32)
    for tile in layout.tiles:
        source = frame[
            tile.read_y0 : tile.read_y1,
            tile.read_x0 : tile.read_x1,
        ]
        enhanced = enhancer.enhance(source)
        expected_shape = (
            (tile.read_y1 - tile.read_y0) * scale,
            (tile.read_x1 - tile.read_x0) * scale,
            3,
        )
        if enhanced.dtype != np.uint8 or enhanced.shape != expected_shape:
            raise ValueError("enhancer tile output shape or dtype violates its scale")
        crop_x0 = (tile.x0 - tile.read_x0) * scale
        crop_y0 = (tile.y0 - tile.read_y0) * scale
        crop_x1 = crop_x0 + (tile.x1 - tile.x0) * scale
        crop_y1 = crop_y0 + (tile.y1 - tile.y0) * scale
        core = enhanced[crop_y0:crop_y1, crop_x0:crop_x1]
        weights = build_blend_weights(tile, layout, scale=scale)
        output_y = slice(tile.y0 * scale, tile.y1 * scale)
        output_x = slice(tile.x0 * scale, tile.x1 * scale)
        accumulation[output_y, output_x] += (
            core.astype(np.float32) * weights[:, :, None]
        )
        total_weight[output_y, output_x] += weights
    if np.any(total_weight <= 0):
        raise ValueError("tile layout left zero-weight output pixels")
    normalized = accumulation / total_weight[:, :, None]
    result = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
    return cast(NDArray[np.uint8], result)
