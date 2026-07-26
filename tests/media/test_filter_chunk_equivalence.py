from __future__ import annotations

from dataclasses import replace

from engvit.config import JobConfig
from engvit.media.filters import build_timing_filter, plan_timing_preparation
from engvit.media.geometry import plan_geometry
from engvit.media.timeline import plan_timeline
from engvit.types import Rational, SourceFrameTiming
from tests.media.test_timeline import config, media_info, video


def test_progressive_filter_freezes_fps_time_base_pts_and_geometry() -> None:
    selected = video()
    geometry = plan_geometry(
        selected,
        JobConfig(
            selected_video_index=2,
            target_width=1280,
            target_height=720,
            fps_policy="source_cfr",
        ),
        model_scale=2,
    )
    source = tuple(
        SourceFrameTiming(
            source_index=index,
            best_effort_pts=index * 1001,
            duration_pts=1001,
            source_time_base=Rational(1, 30000),
            repeat_pict=0,
            interlaced=False,
            top_field_first=None,
        )
        for index in range(3)
    )
    timeline = plan_timeline(media_info(selected), config(), source)
    graph = build_timing_filter(timeline, geometry, selected)
    assert graph.input_options == ("-noautorotate",)
    assert graph.filters[-3:] == (
        "fps=fps=30000/1001:round=near:start_time=0",
        "settb=expr=1001/30000",
        "setpts=N",
    )
    assert graph.continuous is False


def test_interlaced_filter_uses_explicit_mode_parity_and_rate() -> None:
    selected = replace(video(), field_order="tt")
    geometry = plan_geometry(selected, config(), model_scale=2)
    source = (
        SourceFrameTiming(
            source_index=0,
            best_effort_pts=0,
            duration_pts=1001,
            source_time_base=Rational(1, 30000),
            repeat_pict=0,
            interlaced=True,
            top_field_first=True,
        ),
    )
    timeline = plan_timeline(media_info(selected), config(), source)
    graph = build_timing_filter(timeline, geometry, selected)
    assert graph.filters[0] == "bwdif=mode=send_frame:parity=tff:deint=all"
    assert graph.continuous is True
    assert graph.context_before == 2
    assert graph.context_after == 2


def test_chunk_filter_is_the_same_graph_with_only_bounds_changed() -> None:
    selected = video()
    geometry = plan_geometry(selected, config(), model_scale=2)
    source = tuple(
        SourceFrameTiming(
            source_index=index,
            best_effort_pts=index * 1001,
            duration_pts=1001,
            source_time_base=Rational(1, 30000),
            repeat_pict=0,
            interlaced=False,
            top_field_first=None,
        )
        for index in range(5)
    )
    timeline = plan_timeline(media_info(selected), config(), source)
    reference = build_timing_filter(timeline, geometry, selected)
    chunk = reference.for_core(1, 4)
    assert chunk.filters == reference.filters
    assert chunk.core_start == 1
    assert chunk.core_end == 4
    assert reference.core_start == 0
    assert reference.core_end == 5


def test_every_telecine_phase_routes_to_one_full_span_ivtc_mezzanine() -> None:
    selected = replace(video(), field_order="tt")
    observed_filters: set[str] = set()
    for phase in range(5):
        source = tuple(
            SourceFrameTiming(
                source_index=index,
                best_effort_pts=index * 1001,
                duration_pts=1001,
                source_time_base=Rational(1, 30000),
                repeat_pict=1 if index % 5 == phase else 0,
                interlaced=True,
                top_field_first=True,
            )
            for index in range(20)
        )
        preparation = plan_timing_preparation(selected, source)
        assert preparation is not None
        assert preparation.kind == "full_span_ivtc_mezzanine"
        assert preparation.continuous is True
        assert preparation.expected_fps == Rational(24000, 1001)
        observed_filters.add(preparation.video_filter)
    assert observed_filters == {
        "fieldmatch=order=tff:combmatch=full,decimate=cycle=5"
    }
