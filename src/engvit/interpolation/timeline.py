"""Exact half-open interpolation ticks with hard-cut isolation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from engvit.analysis.scenes import Scene
from engvit.canonical import canonical_sha256
from engvit.types import Rational, TimelinePlan


@dataclass(frozen=True)
class InterpolationTick:
    output_index: int
    left_input_index: int
    right_input_index: int | None
    fraction: Rational | None
    scene_id: int


@dataclass(frozen=True)
class InterpolationPlan:
    input_timeline_sha256: str
    input_fps: Rational
    output_fps: Rational
    output_time_base: Rational
    ticks: tuple[InterpolationTick, ...]
    sha256: str


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _scene_map(
    scenes: tuple[Scene, ...],
    frame_count: int,
) -> tuple[int, ...]:
    if not scenes:
        raise ValueError("interpolation requires explicit scene bounds")
    ordered = tuple(sorted(scenes, key=lambda item: item.start_frame))
    cursor = 0
    mapping: list[int] = []
    for scene in ordered:
        if (
            scene.start_frame != cursor
            or scene.end_frame <= scene.start_frame
            or scene.end_frame > frame_count
        ):
            raise ValueError("scenes must be contiguous half-open input bounds")
        mapping.extend([scene.scene_id] * (scene.end_frame - scene.start_frame))
        cursor = scene.end_frame
    if cursor != frame_count or len(mapping) != frame_count:
        raise ValueError("scenes must cover the complete input timeline")
    return tuple(mapping)


def plan_interpolation(
    timeline: TimelinePlan,
    scenes: tuple[Scene, ...],
    target_fps: Rational,
) -> InterpolationPlan:
    """Plan RIFE inputs after CFR normalization without crossing hard cuts."""
    frame_count = len(timeline.output_frames)
    if frame_count < 1:
        raise ValueError("interpolation input timeline is empty")
    if target_fps.numerator <= 0:
        raise ValueError("target_fps must be positive")
    if target_fps.as_float() < timeline.output_fps.as_float():
        raise ValueError("interpolation cannot be used to lower frame rate")
    scene_by_input = _scene_map(scenes, frame_count)
    duration = Fraction(
        frame_count * timeline.output_fps.denominator,
        timeline.output_fps.numerator,
    )
    output_count = _ceil(
        duration * Fraction(target_fps.numerator, target_fps.denominator)
    )
    input_per_output = Fraction(
        timeline.output_fps.numerator * target_fps.denominator,
        timeline.output_fps.denominator * target_fps.numerator,
    )
    ticks: list[InterpolationTick] = []
    for output_index in range(output_count):
        position = output_index * input_per_output
        left = min(position.numerator // position.denominator, frame_count - 1)
        exact = position.denominator == 1
        right = left + 1
        if exact or right >= frame_count:
            ticks.append(
                InterpolationTick(
                    output_index=output_index,
                    left_input_index=left,
                    right_input_index=None,
                    fraction=None,
                    scene_id=scene_by_input[left],
                )
            )
            continue
        if scene_by_input[left] != scene_by_input[right]:
            # Hold the old scene until the exact cut tick. Never synthesize across it.
            ticks.append(
                InterpolationTick(
                    output_index=output_index,
                    left_input_index=left,
                    right_input_index=None,
                    fraction=None,
                    scene_id=scene_by_input[left],
                )
            )
            continue
        offset = position - left
        ticks.append(
            InterpolationTick(
                output_index=output_index,
                left_input_index=left,
                right_input_index=right,
                fraction=Rational(offset.numerator, offset.denominator),
                scene_id=scene_by_input[left],
            )
        )
    provisional = InterpolationPlan(
        input_timeline_sha256=timeline.sha256,
        input_fps=timeline.output_fps,
        output_fps=target_fps,
        output_time_base=Rational(target_fps.denominator, target_fps.numerator),
        ticks=tuple(ticks),
        sha256="",
    )
    return InterpolationPlan(
        input_timeline_sha256=provisional.input_timeline_sha256,
        input_fps=provisional.input_fps,
        output_fps=provisional.output_fps,
        output_time_base=provisional.output_time_base,
        ticks=provisional.ticks,
        sha256=canonical_sha256(provisional, projection="identity"),
    )
