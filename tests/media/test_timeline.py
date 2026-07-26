from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from engvit.canonical import canonical_sha256
from engvit.config import JobConfig
from engvit.media.timeline import bind_raw_frame, plan_timeline
from engvit.types import (
    MediaInfo,
    Rational,
    SourceFrameTiming,
    StreamInfo,
    VideoStreamInfo,
)

NTSC_FPS = Rational(30000, 1001)
VIDEO_TIME_BASE = Rational(1, 30000)


def video(*, fps: Rational = NTSC_FPS) -> VideoStreamInfo:
    return VideoStreamInfo(
        index=2,
        codec_type="video",
        codec_name="ffv1",
        time_base=VIDEO_TIME_BASE,
        start_pts=-1001,
        duration_pts=4004,
        disposition={"default": True},
        language=None,
        metadata={},
        coded_width=640,
        coded_height=360,
        sample_aspect_ratio=Rational(1, 1),
        avg_frame_rate=fps,
        real_frame_rate=fps,
        pixel_format="yuv420p",
        bits_per_raw_sample=8,
        field_order="progressive",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        display_matrix=None,
    )


def media_info(stream: VideoStreamInfo | None = None) -> MediaInfo:
    selected = stream or video()
    audio = StreamInfo(
        index=0,
        codec_type="audio",
        codec_name="flac",
        time_base=Rational(1, 1000),
        start_pts=0,
        duration_pts=1000,
        disposition={"default": True},
        language="eng",
        metadata={},
    )
    return MediaInfo(
        source=Path("source.mkv"),
        source_sha256="a" * 64,
        format_name="matroska",
        duration_seconds=Decimal("1"),
        streams=(audio, selected),
        selected_video_index=selected.index,
    )


def frames(
    points: tuple[int, ...],
    durations: tuple[int, ...],
    *,
    time_base: Rational = VIDEO_TIME_BASE,
) -> tuple[SourceFrameTiming, ...]:
    return tuple(
        SourceFrameTiming(
            source_index=index,
            best_effort_pts=point,
            duration_pts=durations[index],
            source_time_base=time_base,
            repeat_pict=0,
            interlaced=False,
            top_field_first=None,
        )
        for index, point in enumerate(points)
    )


def config(
    policy: str = "source_cfr",
    target: Rational | None = None,
) -> JobConfig:
    return JobConfig(
        selected_video_index=2,
        target_width=1280,
        target_height=720,
        fps_policy=policy,  # type: ignore[arg-type]
        target_fps=target,
    )


def test_source_cfr_preserves_30000_over_1001_and_rebases_negative_pts() -> None:
    source = frames(
        (-1001, 0, 1001, 2002),
        (1001, 1001, 1001, 1001),
    )
    plan = plan_timeline(media_info(), config(), source)
    assert plan.output_fps == Rational(30000, 1001)
    assert plan.output_time_base == Rational(1001, 30000)
    assert tuple(item.output_pts for item in plan.output_frames) == (0, 1, 2, 3)
    assert tuple(item.source_indexes for item in plan.output_frames) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
    assert plan.sha256 == canonical_sha256(
        replace(plan, sha256=""),
        projection="identity",
    )
    assert bind_raw_frame(3, plan) == plan.output_frames[3]


def test_normalize_vfr_to_explicit_25_fps_mapping() -> None:
    vfr_stream = replace(
        video(fps=Rational(25, 1)),
        time_base=Rational(1, 1000),
    )
    source = frames(
        (0, 40, 100),
        (40, 60, 40),
        time_base=Rational(1, 1000),
    )
    plan = plan_timeline(
        media_info(vfr_stream),
        config("normalize_cfr", Rational(25, 1)),
        source,
    )
    assert plan.timing_transform == ("vfr_to_cfr:25/1",)
    assert tuple(item.source_indexes for item in plan.output_frames) == (
        (0,),
        (1,),
        (1,),
        (2,),
    )
    assert plan.output_time_base == Rational(1, 25)


def test_rife_timeline_records_exact_interpolation_fractions() -> None:
    selected = replace(video(fps=Rational(25, 1)), time_base=Rational(1, 1000))
    source = frames((0, 40), (40, 40), time_base=Rational(1, 1000))
    plan = plan_timeline(
        media_info(selected),
        config("rife", Rational(50, 1)),
        source,
    )
    assert plan.output_frames[1].source_indexes == (0, 1)
    assert plan.output_frames[1].interpolation_fraction == Rational(1, 2)
    assert plan.output_frames[2].source_indexes == (1,)
    assert plan.output_frames[3].source_indexes == (1,)


def test_source_cfr_rejects_irregular_timestamps() -> None:
    source = frames(
        (0, 1001, 2500),
        (1001, 1499, 1001),
    )
    with pytest.raises(ValueError, match="not CFR"):
        plan_timeline(media_info(), config(), source)


def test_timeline_rejects_time_base_mismatch() -> None:
    source = frames((0, 1), (1, 1), time_base=Rational(1, 1000))
    with pytest.raises(ValueError, match="time base"):
        plan_timeline(media_info(), config(), source)


def test_bind_raw_frame_rejects_out_of_range_index() -> None:
    source = frames((-1001, 0), (1001, 1001))
    plan = plan_timeline(media_info(), config(), source)
    with pytest.raises(IndexError, match="output frame"):
        bind_raw_frame(2, plan)


def test_timeline_requires_mezzanine_for_repeat_field_cadence() -> None:
    source = list(frames((0, 1001), (1001, 1001)))
    source[1] = replace(source[1], repeat_pict=1, interlaced=True)
    with pytest.raises(ValueError, match="IVTC mezzanine"):
        plan_timeline(media_info(), config(), tuple(source))
