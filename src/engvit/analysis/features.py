"""Versioned, dependency-light proxy-frame feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FEATURE_VERSION = "engvit-proxy-v1"


@dataclass(frozen=True)
class FeatureVector:
    luma_mean: float
    luma_std: float
    chroma_mean: float
    spatial_information: float
    temporal_information: float
    edge_density: float
    noise_score: float
    blocking_score: float
    ringing_score: float
    banding_score: float
    motion: float
    flat_fraction: float
    dark_fraction: float
    highlight_fraction: float
    text_line_score: float
    black: bool
    freeze: bool
    repeat: bool
    face_score: float | None
    face_state: str


def _validate(frame: NDArray[np.uint8]) -> None:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("proxy frame must be an HxWx3 uint8 RGB array")
    if frame.shape[0] < 8 or frame.shape[1] < 8:
        raise ValueError("proxy RGB frame must be at least 8x8")


def _luma(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    rgb = frame.astype(np.float32) / np.float32(255.0)
    result = (
        np.float32(0.2126) * rgb[:, :, 0]
        + np.float32(0.7152) * rgb[:, :, 1]
        + np.float32(0.0722) * rgb[:, :, 2]
    )
    return cast(NDArray[np.float32], result)


def _blocking_score(luma: NDArray[np.float32]) -> float:
    vertical = np.abs(np.diff(luma, axis=1))
    horizontal = np.abs(np.diff(luma, axis=0))
    boundary_values: list[NDArray[np.float32]] = []
    if vertical.shape[1] >= 8:
        boundary_values.append(vertical[:, 7::8])
    if horizontal.shape[0] >= 8:
        boundary_values.append(horizontal[7::8, :])
    if not boundary_values:
        return 0.0
    boundary = float(np.mean([float(np.mean(item)) for item in boundary_values]))
    baseline = (float(np.mean(vertical)) + float(np.mean(horizontal))) / 2
    return max(0.0, boundary - baseline)


def extract_features(
    frame: NDArray[np.uint8],
    previous: NDArray[np.uint8] | None,
) -> FeatureVector:
    """Extract transparent heuristics; unavailable face inference stays explicit."""
    _validate(frame)
    if previous is not None:
        _validate(previous)
        if previous.shape != frame.shape:
            raise ValueError("consecutive proxy RGB frames must have identical shape")

    luma = _luma(frame)
    dx = np.diff(luma, axis=1, append=luma[:, -1:])
    dy = np.diff(luma, axis=0, append=luma[-1:, :])
    gradient = np.sqrt(dx * dx + dy * dy)
    spatial_information = float(np.std(gradient))
    edge_density = float(np.mean(gradient > np.float32(0.08)))
    flat_fraction = float(np.mean(gradient < np.float32(2 / 255)))

    neighbor_average = (
        np.roll(luma, 1, axis=0)
        + np.roll(luma, -1, axis=0)
        + np.roll(luma, 1, axis=1)
        + np.roll(luma, -1, axis=1)
    ) / np.float32(4)
    residual = np.abs(luma - neighbor_average)
    noise_score = float(np.median(residual))
    second_x = np.abs(np.diff(dx, axis=1, prepend=dx[:, :1]))
    second_y = np.abs(np.diff(dy, axis=0, prepend=dy[:1, :]))
    ringing_score = float(np.mean(np.maximum(second_x, second_y)))

    horizontal_steps = np.abs(np.diff(luma, axis=1))
    vertical_steps = np.abs(np.diff(luma, axis=0))
    banding_score = float(
        (
            np.mean(horizontal_steps < np.float32(1 / 255))
            + np.mean(vertical_steps < np.float32(1 / 255))
        )
        / 2
    )
    axis_edges = np.maximum(np.abs(dx), np.abs(dy))
    text_line_score = float(np.mean(axis_edges > np.float32(0.15)))

    if previous is None:
        temporal_information = 0.0
        motion = 0.0
        repeat = False
    else:
        previous_luma = _luma(previous)
        delta = np.abs(luma - previous_luma)
        temporal_information = float(np.std(delta))
        motion = float(np.mean(delta))
        repeat = bool(np.array_equal(frame, previous))

    rgb_float = frame.astype(np.float32) / np.float32(255.0)
    chroma_mean = float(np.mean(np.max(rgb_float, axis=2) - np.min(rgb_float, axis=2)))
    luma_mean = float(np.mean(luma))
    return FeatureVector(
        luma_mean=luma_mean,
        luma_std=float(np.std(luma)),
        chroma_mean=chroma_mean,
        spatial_information=spatial_information,
        temporal_information=temporal_information,
        edge_density=edge_density,
        noise_score=noise_score,
        blocking_score=_blocking_score(luma),
        ringing_score=ringing_score,
        banding_score=banding_score,
        motion=motion,
        flat_fraction=flat_fraction,
        dark_fraction=float(np.mean(luma < np.float32(16 / 255))),
        highlight_fraction=float(np.mean(luma > np.float32(235 / 255))),
        text_line_score=text_line_score,
        black=luma_mean < 16 / 255,
        freeze=previous is not None and motion < 1 / 255,
        repeat=repeat,
        face_score=None,
        face_state="NOT_EVALUATED",
    )
