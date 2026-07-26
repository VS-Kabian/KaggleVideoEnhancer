"""Exact odd-dimension tile layout with explicit read context and overlaps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileRegion:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    read_x0: int
    read_y0: int
    read_x1: int
    read_y1: int
    overlap_left: int
    overlap_right: int
    overlap_top: int
    overlap_bottom: int


@dataclass(frozen=True)
class TileLayout:
    width: int
    height: int
    tile_size: int
    overlap: int
    context: int
    x_positions: tuple[int, ...]
    y_positions: tuple[int, ...]
    tiles: tuple[TileRegion, ...]


def _positions(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    step = tile_size - overlap
    result = [0]
    while result[-1] + tile_size < length:
        position = min(result[-1] + step, length - tile_size)
        if position == result[-1]:
            break
        result.append(position)
    return tuple(result)


def plan_layout(
    *,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    context: int,
) -> TileLayout:
    """Anchor both edges and cover every input pixel with overlapping cores."""
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("image and tile dimensions must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile size")
    if context < 0:
        raise ValueError("context must be non-negative")
    x_positions = _positions(width, tile_size, overlap)
    y_positions = _positions(height, tile_size, overlap)
    tiles: list[TileRegion] = []
    for y_index, y0 in enumerate(y_positions):
        y1 = min(height, y0 + tile_size)
        top = (
            max(0, y_positions[y_index - 1] + tile_size - y0)
            if y_index > 0
            else 0
        )
        bottom = (
            max(0, y1 - y_positions[y_index + 1])
            if y_index + 1 < len(y_positions)
            else 0
        )
        for x_index, x0 in enumerate(x_positions):
            x1 = min(width, x0 + tile_size)
            left = (
                max(0, x_positions[x_index - 1] + tile_size - x0)
                if x_index > 0
                else 0
            )
            right = (
                max(0, x1 - x_positions[x_index + 1])
                if x_index + 1 < len(x_positions)
                else 0
            )
            tiles.append(
                TileRegion(
                    index=len(tiles),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    read_x0=max(0, x0 - context),
                    read_y0=max(0, y0 - context),
                    read_x1=min(width, x1 + context),
                    read_y1=min(height, y1 + context),
                    overlap_left=left,
                    overlap_right=right,
                    overlap_top=top,
                    overlap_bottom=bottom,
                )
            )
    return TileLayout(
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        context=context,
        x_positions=x_positions,
        y_positions=y_positions,
        tiles=tuple(tiles),
    )

