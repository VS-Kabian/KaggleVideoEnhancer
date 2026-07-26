from __future__ import annotations

from pathlib import Path

from engvit.types import (
    OutputFrameSpec,
    Rational,
    SourceFrameTiming,
    TimelinePlan,
)

DEFAULT_FPS = Rational(10, 1)


def timeline(frame_count: int, fps: Rational = DEFAULT_FPS) -> TimelinePlan:
    source = tuple(
        SourceFrameTiming(
            source_index=index,
            best_effort_pts=index,
            duration_pts=1,
            source_time_base=Rational(fps.denominator, fps.numerator),
            repeat_pict=0,
            interlaced=False,
            top_field_first=None,
        )
        for index in range(frame_count)
    )
    output = tuple(
        OutputFrameSpec(
            output_index=index,
            output_pts=index,
            output_duration=1,
            source_indexes=(index,),
            interpolation_fraction=None,
        )
        for index in range(frame_count)
    )
    return TimelinePlan(
        source_time_base=Rational(fps.denominator, fps.numerator),
        output_time_base=Rational(fps.denominator, fps.numerator),
        output_fps=fps,
        timing_transform=(f"source_cfr:{fps.numerator}/{fps.denominator}",),
        source_frames=source,
        output_frames=output,
        sha256="b" * 64,
    )


def unused_path() -> Path:
    return Path("unused")
