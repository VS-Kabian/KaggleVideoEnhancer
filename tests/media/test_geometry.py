from __future__ import annotations

import pytest

from engvit.config import JobConfig
from engvit.media.geometry import plan_geometry
from engvit.types import Rational, VideoStreamInfo

SQUARE_PIXELS = Rational(1, 1)


def stream(
    *,
    width: int,
    height: int,
    sar: Rational = SQUARE_PIXELS,
    matrix: tuple[float, ...] | None = None,
) -> VideoStreamInfo:
    return VideoStreamInfo(
        index=0,
        codec_type="video",
        codec_name="ffv1",
        time_base=Rational(1, 1000),
        start_pts=0,
        duration_pts=1000,
        disposition={"default": True},
        language=None,
        metadata={},
        coded_width=width,
        coded_height=height,
        sample_aspect_ratio=sar,
        avg_frame_rate=Rational(30, 1),
        real_frame_rate=Rational(30, 1),
        pixel_format="yuv420p",
        bits_per_raw_sample=8,
        field_order="progressive",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        display_matrix=matrix,
    )


def job(width: int = 3840, height: int = 2160) -> JobConfig:
    return JobConfig(
        selected_video_index=0,
        target_width=width,
        target_height=height,
        fps_policy="source_cfr",
    )


def test_plan_geometry_uses_native_x2_without_final_resize() -> None:
    plan = plan_geometry(stream(width=1920, height=1080), job(), model_scale=2)
    assert plan.oriented_size == (1920, 1080)
    assert plan.target_size == (3840, 2160)
    assert plan.model_size == (3840, 2160)
    assert plan.final_resize is None
    assert plan.output_sar == Rational(1, 1)


def test_plan_geometry_applies_sar_then_rotation_once() -> None:
    rotate_90 = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    plan = plan_geometry(
        stream(
            width=720,
            height=576,
            sar=Rational(16, 15),
            matrix=rotate_90,
        ),
        job(),
        model_scale=2,
    )
    assert plan.oriented_size == (576, 768)
    assert plan.target_size == (1620, 2160)
    assert plan.model_size == (1152, 1536)
    assert plan.final_resize == "lanczos"
    assert plan.pixel_transform == ("scale_sar:768x576", "rotate:90", "setsar:1")


def test_plan_geometry_rejects_non_orthogonal_display_matrix() -> None:
    skew = (1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="orthogonal"):
        plan_geometry(stream(width=640, height=480, matrix=skew), job(), model_scale=2)
