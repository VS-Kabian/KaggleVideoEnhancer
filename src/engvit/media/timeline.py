"""Exact rational planning from source frame timing to normalized CFR output."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from fractions import Fraction

from engvit.canonical import canonical_sha256
from engvit.config import JobConfig
from engvit.types import (
    MediaInfo,
    OutputFrameSpec,
    Rational,
    SourceFrameTiming,
    TimelinePlan,
    VideoStreamInfo,
)


def _selected_video(media: MediaInfo) -> VideoStreamInfo:
    selected = next(
        (
            stream
            for stream in media.streams
            if stream.index == media.selected_video_index
            and isinstance(stream, VideoStreamInfo)
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected video stream is absent")
    return selected


def _validate_frames(
    video: VideoStreamInfo,
    frames: tuple[SourceFrameTiming, ...],
) -> Rational:
    if not frames:
        raise ValueError("source timing contains no frames")
    if video.time_base is None:
        raise ValueError("selected video has no usable time base")
    previous: int | None = None
    for expected_index, frame in enumerate(frames):
        if frame.source_index != expected_index:
            raise ValueError("source frame indexes must be contiguous from zero")
        if frame.source_time_base != video.time_base:
            raise ValueError("source frame time base does not match selected video")
        if frame.duration_pts <= 0:
            raise ValueError("source frame durations must be positive")
        if previous is not None and frame.best_effort_pts <= previous:
            raise ValueError("source frame timestamps must be strictly increasing")
        previous = frame.best_effort_pts
    return video.time_base


def _nominal_ticks(fps: Rational, time_base: Rational) -> Fraction:
    return Fraction(
        fps.denominator * time_base.denominator,
        fps.numerator * time_base.numerator,
    )


def _require_cfr(
    frames: tuple[SourceFrameTiming, ...],
    fps: Rational,
    time_base: Rational,
) -> None:
    expected = _nominal_ticks(fps, time_base)
    origin = frames[0].best_effort_pts
    for index, frame in enumerate(frames):
        expected_pts = Fraction(origin) + index * expected
        if abs(Fraction(frame.best_effort_pts) - expected_pts) > 1:
            raise ValueError("source timing is not CFR at its declared frame rate")
        if abs(Fraction(frame.duration_pts) - expected) > 1:
            raise ValueError("source timing is not CFR at its declared frame rate")


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _normalized_frames(
    source_frames: tuple[SourceFrameTiming, ...],
    output_fps: Rational,
    *,
    interpolate: bool,
) -> tuple[OutputFrameSpec, ...]:
    time_base = source_frames[0].source_time_base
    first_pts = source_frames[0].best_effort_pts
    end_pts = source_frames[-1].best_effort_pts + source_frames[-1].duration_pts
    duration_seconds = Fraction(
        (end_pts - first_pts) * time_base.numerator,
        time_base.denominator,
    )
    output_count = _ceil(
        duration_seconds * Fraction(output_fps.numerator, output_fps.denominator)
    )
    period_ticks = _nominal_ticks(output_fps, time_base)
    source_points = tuple(frame.best_effort_pts for frame in source_frames)
    output: list[OutputFrameSpec] = []
    for output_index in range(output_count):
        source_position = Fraction(first_pts) + output_index * period_ticks
        left = bisect_right(source_points, source_position) - 1
        left = min(max(left, 0), len(source_frames) - 1)
        exact = source_position == source_points[left]
        right = left + 1
        if interpolate and not exact and right < len(source_frames):
            span = source_points[right] - source_points[left]
            offset = source_position - source_points[left]
            fraction = Rational(offset.numerator, offset.denominator * span)
            source_indexes: tuple[int, ...] = (left, right)
        else:
            fraction = None
            source_indexes = (left,)
        output.append(
            OutputFrameSpec(
                output_index=output_index,
                output_pts=output_index,
                output_duration=1,
                source_indexes=source_indexes,
                interpolation_fraction=fraction,
            )
        )
    return tuple(output)


def plan_timeline(
    media: MediaInfo,
    config: JobConfig,
    source_frames: tuple[SourceFrameTiming, ...],
) -> TimelinePlan:
    """Freeze a complete source-to-output mapping under an explicit FPS policy."""
    video = _selected_video(media)
    time_base = _validate_frames(video, source_frames)
    if config.selected_video_index != media.selected_video_index:
        raise ValueError("job selected video index does not match probed media")
    if any(frame.repeat_pict > 0 for frame in source_frames):
        raise ValueError(
            "telecine timing requires a full-span continuous IVTC mezzanine"
        )

    if config.fps_policy == "source_cfr":
        if video.avg_frame_rate is None:
            raise ValueError("source CFR policy requires a declared average frame rate")
        output_fps = video.avg_frame_rate
        _require_cfr(source_frames, output_fps, time_base)
        output_frames = tuple(
            OutputFrameSpec(
                output_index=index,
                output_pts=index,
                output_duration=1,
                source_indexes=(index,),
                interpolation_fraction=None,
            )
            for index in range(len(source_frames))
        )
        transforms = (f"source_cfr:{output_fps.numerator}/{output_fps.denominator}",)
    else:
        if config.target_fps is None:
            raise ValueError("normalized timing requires target_fps")
        output_fps = config.target_fps
        output_frames = _normalized_frames(
            source_frames,
            output_fps,
            interpolate=config.fps_policy == "rife",
        )
        prefix = "rife" if config.fps_policy == "rife" else "vfr_to_cfr"
        transforms = (f"{prefix}:{output_fps.numerator}/{output_fps.denominator}",)

    provisional = TimelinePlan(
        source_time_base=time_base,
        output_time_base=Rational(output_fps.denominator, output_fps.numerator),
        output_fps=output_fps,
        timing_transform=transforms,
        source_frames=source_frames,
        output_frames=output_frames,
        sha256="",
    )
    return replace(
        provisional,
        sha256=canonical_sha256(provisional, projection="identity"),
    )


def bind_raw_frame(output_index: int, plan: TimelinePlan) -> OutputFrameSpec:
    """Bind one raw RGB frame index to its frozen timeline record."""
    if output_index < 0 or output_index >= len(plan.output_frames):
        raise IndexError("output frame index is outside the timeline")
    frame = plan.output_frames[output_index]
    if frame.output_index != output_index:
        raise ValueError("timeline output indexes are not canonical")
    return frame
