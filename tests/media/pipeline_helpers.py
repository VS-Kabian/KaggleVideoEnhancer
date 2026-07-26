from __future__ import annotations

import hashlib
import subprocess
from decimal import Decimal
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from engvit.config import JobConfig
from engvit.media.filters import build_timing_filter
from engvit.media.geometry import plan_geometry
from engvit.media.timeline import plan_timeline
from engvit.types import (
    EncoderConfig,
    MediaInfo,
    Rational,
    SourceFrameTiming,
    StreamInfo,
    TimelinePlan,
    VideoStreamInfo,
)


def ffmpeg_path() -> Path:
    return Path(imageio_ffmpeg.get_ffmpeg_exe())


def make_video(path: Path, *, frames: int = 6) -> None:
    subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=32x18:rate=10:duration={frames / 10}",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-bitexact",
            str(path),
        ],
        check=True,
    )


def video_stream() -> VideoStreamInfo:
    return VideoStreamInfo(
        index=0,
        codec_type="video",
        codec_name="ffv1",
        time_base=Rational(1, 10),
        start_pts=0,
        duration_pts=6,
        disposition={"default": True},
        language=None,
        metadata={},
        coded_width=32,
        coded_height=18,
        sample_aspect_ratio=Rational(1, 1),
        avg_frame_rate=Rational(10, 1),
        real_frame_rate=Rational(10, 1),
        pixel_format="yuv420p",
        bits_per_raw_sample=8,
        field_order="progressive",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        display_matrix=None,
    )


def pipeline_contract(source: Path, frame_count: int = 6) -> tuple[TimelinePlan, object]:
    selected = video_stream()
    frames = tuple(
        SourceFrameTiming(
            source_index=index,
            best_effort_pts=index,
            duration_pts=1,
            source_time_base=Rational(1, 10),
            repeat_pict=0,
            interlaced=False,
            top_field_first=None,
        )
        for index in range(frame_count)
    )
    media = MediaInfo(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        format_name="matroska",
        duration_seconds=Decimal(str(frame_count / 10)),
        streams=(selected,),
        selected_video_index=0,
    )
    job = JobConfig(
        selected_video_index=0,
        target_width=64,
        target_height=36,
        fps_policy="source_cfr",
    )
    timeline = plan_timeline(media, job, frames)
    geometry = plan_geometry(selected, job, model_scale=2)
    return timeline, build_timing_filter(timeline, geometry, selected)


def encoder_config() -> EncoderConfig:
    return EncoderConfig(
        ffmpeg_sha256=hashlib.sha256(ffmpeg_path().read_bytes()).hexdigest(),
        encoder="libx264",
        hardware_device=None,
        codec="h264",
        pixel_format="yuv420p",
        rate_control={"mode": "crf", "crf": 18},
        preset="veryfast",
        gop=3,
        b_frames=0,
        output_fps=Rational(10, 1),
        output_time_base=Rational(1, 10),
        color={
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        },
        self_test_sha256="e" * 64,
    )


def rgb_frames(count: int, *, offset: int = 0) -> tuple[np.ndarray, ...]:
    return tuple(
        np.full((36, 64, 3), (index + offset) * 20, dtype=np.uint8)
        for index in range(count)
    )


def unused_stream() -> StreamInfo:
    return StreamInfo(
        index=1,
        codec_type="audio",
        codec_name="flac",
        time_base=Rational(1, 48000),
        start_pts=0,
        duration_pts=48000,
        disposition={"default": True},
        language="eng",
        metadata={"title": "private canary"},
    )
