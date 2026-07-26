"""Self-contained small Phase 0 job used by notebook and CI smoke checks."""

from __future__ import annotations

import hashlib
import subprocess
from decimal import Decimal
from pathlib import Path

from engvit.canonical import canonical_sha256
from engvit.config import JobConfig
from engvit.kaggle import baseline_encoder_config
from engvit.media.color import classify_color
from engvit.media.filters import build_timing_filter
from engvit.media.geometry import plan_geometry
from engvit.media.timeline import plan_timeline
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from engvit.orchestration.coordinator import Coordinator
from engvit.paths import create_job_paths
from engvit.pipeline import Phase0Result, execute_lanczos_job
from engvit.types import (
    MediaInfo,
    Rational,
    SourceFrameTiming,
    VideoStreamInfo,
)


def run_phase0_smoke(root: Path, ffmpeg_path: Path) -> Phase0Result:
    """Generate six frames and execute a real two-chunk Lanczos/encode job."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "smoke-source.mkv"
    subprocess.run(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x18:rate=10:duration=0.6",
            "-frames:v",
            "6",
            "-c:v",
            "ffv1",
            "-bitexact",
            str(source),
        ],
        check=True,
        shell=False,
    )
    video = VideoStreamInfo(
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
    media = MediaInfo(
        source=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        format_name="matroska",
        duration_seconds=Decimal("0.6"),
        streams=(video,),
        selected_video_index=0,
    )
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
        for index in range(6)
    )
    job = JobConfig(
        selected_video_index=0,
        target_width=64,
        target_height=36,
        fps_policy="source_cfr",
    )
    timeline = plan_timeline(media, job, frames)
    geometry = plan_geometry(video, job, model_scale=2)
    graph = build_timing_filter(timeline, geometry, video)
    color = classify_color(video)
    if color.output_color is None:
        raise RuntimeError("smoke color classification unexpectedly failed")
    encoder = baseline_encoder_config(
        ffmpeg_path=ffmpeg_path,
        fps=timeline.output_fps,
        color=color.output_color,
        crf=18,
        preset="veryfast",
    )
    identity = canonical_sha256(
        {
            "pipeline": "phase0-lanczos-v1",
            "timeline": timeline,
            "geometry": geometry,
            "encoder": encoder,
        },
        projection="identity",
    )
    chunks = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256=identity,
        ),
    )
    paths = create_job_paths(root / "jobs", f"smoke-{identity[:12]}")
    coordinator = Coordinator(
        manifest_path=paths.artifacts / "manifest.json",
        segments_root=paths.segments,
        chunks=chunks,
        job_identity_sha256=identity,
    )
    return execute_lanczos_job(
        source=source,
        video_stream_index=0,
        timeline=timeline,
        filter_graph=graph,
        chunks=chunks,
        encoder=encoder,
        output_size=geometry.target_size,
        ffmpeg_path=ffmpeg_path,
        coordinator=coordinator,
        output_path=paths.artifacts / "enhanced-video.mkv",
        worker_id="smoke-worker",
    )
