"""Single-application SAR/orientation geometry planning."""

from __future__ import annotations

from math import isclose
from typing import Literal

from engvit.config import JobConfig
from engvit.types import GeometryPlan, Rational, VideoStreamInfo


def _even_floor(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _orientation(
    matrix: tuple[float, ...] | None,
) -> tuple[str | None, bool]:
    if matrix is None:
        return None, False
    if len(matrix) != 9:
        raise ValueError("display matrix must be a 3x3 orthogonal transform")
    a, b, _, d, e, _, _, _, _ = matrix
    scale = max(abs(a), abs(b), abs(d), abs(e))
    if scale == 0:
        raise ValueError("display matrix is not an orthogonal transform")
    normalized = tuple(value / scale for value in (a, b, d, e))
    patterns: tuple[tuple[tuple[float, ...], str | None, bool], ...] = (
        ((1, 0, 0, 1), None, False),
        ((0, -1, 1, 0), "rotate:90", True),
        ((-1, 0, 0, -1), "rotate:180", False),
        ((0, 1, -1, 0), "rotate:270", True),
        ((-1, 0, 0, 1), "hflip", False),
        ((1, 0, 0, -1), "vflip", False),
        ((0, 1, 1, 0), "transpose", True),
        ((0, -1, -1, 0), "transverse", True),
    )
    for expected, transform, swaps_axes in patterns:
        if all(
            isclose(value, target, abs_tol=1e-6)
            for value, target in zip(normalized, expected, strict=True)
        ):
            return transform, swaps_axes
    raise ValueError("display matrix is not a supported orthogonal transform")


def plan_geometry(
    video: VideoStreamInfo,
    job: JobConfig,
    *,
    model_scale: int,
) -> GeometryPlan:
    """Plan square-pixel orientation, native model output, and final fit."""
    if video.coded_width <= 0 or video.coded_height <= 0:
        raise ValueError("coded video dimensions must be positive")
    if model_scale < 1:
        raise ValueError("model_scale must be positive")

    sar = video.sample_aspect_ratio or Rational(1, 1)
    square_width = _even_floor(
        video.coded_width * sar.numerator / sar.denominator
    )
    square_height = video.coded_height
    transforms: list[str] = []
    if sar != Rational(1, 1):
        transforms.append(f"scale_sar:{square_width}x{square_height}")

    orientation, swaps_axes = _orientation(video.display_matrix)
    if orientation is not None:
        transforms.append(orientation)
    oriented = (
        (square_height, square_width)
        if swaps_axes
        else (square_width, square_height)
    )
    transforms.append("setsar:1")

    fit = min(job.target_width / oriented[0], job.target_height / oriented[1])
    target = (_even_floor(oriented[0] * fit), _even_floor(oriented[1] * fit))
    model_size = (oriented[0] * model_scale, oriented[1] * model_scale)
    final_resize: Literal["lanczos"] | None = (
        None if model_size == target else "lanczos"
    )
    return GeometryPlan(
        coded_size=(video.coded_width, video.coded_height),
        oriented_size=oriented,
        target_size=target,
        model_size=model_size,
        input_sar=video.sample_aspect_ratio,
        output_sar=Rational(1, 1),
        pixel_transform=tuple(transforms),
        final_resize=final_resize,
    )
