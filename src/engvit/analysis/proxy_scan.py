"""Bounded-memory scan of a full normalized output timeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from engvit.analysis.features import FEATURE_VERSION, FeatureVector, extract_features
from engvit.canonical import canonical_sha256
from engvit.types import TimelinePlan


@dataclass(frozen=True)
class ProxyFrame:
    output_index: int
    rgb: NDArray[np.uint8]


@dataclass(frozen=True)
class ProxyRow(FeatureVector):
    output_index: int


@dataclass(frozen=True)
class ProxyScan:
    timeline_sha256: str
    output_frame_count: int
    sample_every_frames: int
    feature_version: str
    rows: tuple[ProxyRow, ...]
    rows_sha256: str


def _cadence(plan: TimelinePlan, max_samples_per_second: int) -> int:
    if max_samples_per_second <= 0:
        raise ValueError("max_samples_per_second must be positive")
    ratio = Fraction(
        plan.output_fps.numerator,
        plan.output_fps.denominator * max_samples_per_second,
    )
    return max(1, -(-ratio.numerator // ratio.denominator))


def _row(
    frame: ProxyFrame,
    previous: NDArray[np.uint8] | None,
) -> ProxyRow:
    features = extract_features(frame.rgb, previous)
    return ProxyRow(output_index=frame.output_index, **asdict(features))


def scan_proxy(
    frames: Iterable[ProxyFrame],
    timeline: TimelinePlan,
    *,
    max_samples_per_second: int = 2,
) -> ProxyScan:
    """Scan a complete ordered frame iterator while retaining cadence rows only."""
    cadence = _cadence(timeline, max_samples_per_second)
    rows: list[ProxyRow] = []
    previous_sample: NDArray[np.uint8] | None = None
    last: ProxyFrame | None = None
    expected_count = len(timeline.output_frames)
    observed_count = 0
    for expected_index, item in enumerate(frames):
        if not isinstance(item, ProxyFrame):
            raise ValueError("frames must contain ProxyFrame values")
        if item.output_index != expected_index:
            raise ValueError("proxy frame coordinates must be contiguous from zero")
        if item.output_index >= expected_count:
            raise ValueError("proxy frame coordinate exceeds timeline")
        observed_count += 1
        last = ProxyFrame(item.output_index, item.rgb.copy())
        if item.output_index % cadence == 0:
            rows.append(_row(item, previous_sample))
            previous_sample = item.rgb.copy()
    if observed_count != expected_count:
        raise ValueError("proxy frame coordinates must cover the complete timeline")
    if last is not None and (not rows or rows[-1].output_index != last.output_index):
        rows.append(_row(last, previous_sample))
    rows_tuple = tuple(rows)
    return ProxyScan(
        timeline_sha256=timeline.sha256,
        output_frame_count=expected_count,
        sample_every_frames=cadence,
        feature_version=FEATURE_VERSION,
        rows=rows_tuple,
        rows_sha256=canonical_sha256(rows_tuple, projection="identity"),
    )
