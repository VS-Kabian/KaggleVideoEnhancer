"""Canonical FFmpeg timing/geometry filter descriptions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from engvit.types import (
    GeometryPlan,
    Rational,
    SourceFrameTiming,
    TimelinePlan,
    VideoStreamInfo,
)


@dataclass(frozen=True)
class TimingPreparation:
    """A full-span lossless timing pass required before normal chunk planning."""

    kind: str
    input_options: tuple[str, ...]
    video_filter: str
    output_options: tuple[str, ...]
    expected_fps: Rational
    continuous: bool


@dataclass(frozen=True)
class FilterGraph:
    input_options: tuple[str, ...]
    filters: tuple[str, ...]
    continuous: bool
    context_before: int
    context_after: int
    core_start: int
    core_end: int

    def for_core(self, start: int, end: int) -> FilterGraph:
        if start < 0 or end <= start or end > self.core_end:
            raise ValueError("filter core must be a non-empty contained range")
        return replace(self, core_start=start, core_end=end)


def plan_timing_preparation(
    video: VideoStreamInfo,
    source_frames: tuple[SourceFrameTiming, ...],
) -> TimingPreparation | None:
    """Require continuous IVTC when repeat-field cadence is present."""
    if not any(frame.repeat_pict > 0 for frame in source_frames):
        return None
    if video.avg_frame_rate is None:
        raise ValueError("telecine preparation requires a declared input frame rate")
    order = "bff" if video.field_order in {"bb", "bt"} else "tff"
    expected_fps = Rational(
        video.avg_frame_rate.numerator * 4,
        video.avg_frame_rate.denominator * 5,
    )
    return TimingPreparation(
        kind="full_span_ivtc_mezzanine",
        input_options=("-noautorotate",),
        video_filter=f"fieldmatch=order={order}:combmatch=full,decimate=cycle=5",
        output_options=(
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-vsync",
            "0",
        ),
        expected_fps=expected_fps,
        continuous=True,
    )


def _geometry_filters(geometry: GeometryPlan) -> tuple[str, ...]:
    filters: list[str] = []
    for transform in geometry.pixel_transform:
        if transform.startswith("scale_sar:"):
            size = transform.split(":", 1)[1]
            width, height = size.split("x", 1)
            filters.append(f"scale=w={width}:h={height}:flags=lanczos")
        elif transform == "rotate:90":
            filters.append("transpose=clock")
        elif transform == "rotate:180":
            filters.extend(("hflip", "vflip"))
        elif transform == "rotate:270":
            filters.append("transpose=cclock")
        elif transform in {"hflip", "vflip"}:
            filters.append(transform)
        elif transform == "transpose":
            filters.append("transpose=clock_flip")
        elif transform == "transverse":
            filters.append("transpose=cclock_flip")
        elif transform == "setsar:1":
            filters.append("setsar=1")
        else:
            raise ValueError(f"unsupported pixel transform: {transform}")
    return tuple(filters)


def _deinterlace_filter(
    video: VideoStreamInfo,
    timeline: TimelinePlan,
) -> tuple[str, ...]:
    interlaced = any(frame.interlaced for frame in timeline.source_frames)
    declared = video.field_order not in (None, "progressive", "unknown")
    if not interlaced and not declared:
        return ()
    if any(frame.repeat_pict > 0 for frame in timeline.source_frames):
        raise ValueError("telecine requires a dedicated continuous IVTC timeline")
    if video.field_order in {"tt", "tb"}:
        parity = "tff"
    elif video.field_order in {"bb", "bt"}:
        parity = "bff"
    else:
        parities = {
            frame.top_field_first
            for frame in timeline.source_frames
            if frame.interlaced and frame.top_field_first is not None
        }
        if len(parities) != 1:
            raise ValueError("interlaced source has ambiguous field parity")
        parity = "tff" if parities.pop() else "bff"
    return (f"bwdif=mode=send_frame:parity={parity}:deint=all",)


def build_timing_filter(
    plan: TimelinePlan,
    geometry: GeometryPlan,
    video: VideoStreamInfo,
) -> FilterGraph:
    """Build one reusable continuous filter graph with explicit time semantics."""
    temporal = _deinterlace_filter(video, plan)
    rate = f"{plan.output_fps.numerator}/{plan.output_fps.denominator}"
    time_base = (
        f"{plan.output_time_base.numerator}/{plan.output_time_base.denominator}"
    )
    filters = (
        temporal
        + _geometry_filters(geometry)
        + (
            f"fps=fps={rate}:round=near:start_time=0",
            f"settb=expr={time_base}",
            "setpts=N",
        )
    )
    continuous = bool(temporal)
    context = 2 if continuous else 0
    return FilterGraph(
        input_options=("-noautorotate",),
        filters=filters,
        continuous=continuous,
        context_before=context,
        context_after=context,
        core_start=0,
        core_end=len(plan.output_frames),
    )
