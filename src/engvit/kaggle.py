"""High-level, fail-closed Kaggle Phase 0 preparation and execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engvit.canonical import canonical_bytes, canonical_sha256
from engvit.config import AppConfig, JobConfig
from engvit.environment import EnvironmentLock, capture_environment
from engvit.media.color import ColorDecision, classify_color
from engvit.media.filters import (
    FilterGraph,
    plan_timing_preparation,
)
from engvit.media.frame_probe import stream_source_timing
from engvit.media.geometry import plan_geometry
from engvit.media.probe import probe_media
from engvit.media.selection import (
    DatasetRoot,
    MediaCandidate,
    Selection,
    discover_media,
    persist_selection,
    select_media,
)
from engvit.media.timeline import plan_timeline
from engvit.orchestration.atomic import AtomicArtifactWriter
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from engvit.orchestration.coordinator import Coordinator
from engvit.paths import create_job_paths
from engvit.pipeline import Phase0Result, execute_lanczos_job
from engvit.types import (
    ChunkSpec,
    EncoderConfig,
    FPSPolicy,
    GeometryPlan,
    JobPaths,
    MediaInfo,
    Rational,
    TimelinePlan,
    VideoStreamInfo,
)


class KagglePhase0Request(BaseModel):
    """User-editable, non-executable Kaggle job request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_handle: str
    dataset_version: str
    dataset_root: Path
    relative_video_path: str
    output_root: Path
    job_id: str = "engvit-job"
    selected_video_index: int | None = Field(default=None, ge=0)
    target_width: int = Field(default=3840, ge=2, le=3840)
    target_height: int = Field(default=2160, ge=2, le=2160)
    fps_policy: FPSPolicy = "source_cfr"
    target_fps: Rational | None = None
    chunk_frames: int = Field(default=300, ge=1)
    crf: int = Field(default=18, ge=0, le=51)
    preset: Literal[
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ] = "medium"


@dataclass(frozen=True)
class PreparedPhase0:
    request: KagglePhase0Request
    paths: JobPaths
    selection: Selection
    media: MediaInfo
    video: VideoStreamInfo
    color: ColorDecision
    timeline: TimelinePlan
    geometry: GeometryPlan
    filter_graph: FilterGraph
    encoder: EncoderConfig
    chunks: tuple[ChunkSpec, ...]
    environment: EnvironmentLock
    execution_identity_sha256: str
    coordinator: Coordinator
    ffmpeg_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_indexes(path: Path, ffprobe_path: Path) -> tuple[int, ...]:
    completed = subprocess.run(
        [
            str(ffprobe_path),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        raise ValueError("FFprobe did not return a streams array")
    indexes = tuple(
        int(stream["index"])
        for stream in streams
        if isinstance(stream, dict)
        and stream.get("codec_type") == "video"
        and isinstance(stream.get("index"), int)
    )
    if not indexes:
        raise ValueError("selected media contains no video stream")
    return indexes


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
        raise ValueError("probed selection has no selected video")
    return selected


def baseline_encoder_config(
    *,
    ffmpeg_path: Path,
    fps: Rational,
    color: dict[str, str],
    crf: int,
    preset: str,
) -> EncoderConfig:
    """Freeze the tested software H.264 baseline; release qualification is separate."""
    ffmpeg_sha256 = _sha256(ffmpeg_path)
    gop = max(1, round(fps.as_float() * 2))
    contract = {
        "ffmpeg_sha256": ffmpeg_sha256,
        "encoder": "libx264",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "crf": crf,
        "preset": preset,
        "gop": gop,
        "b_frames": 0,
        "fps": fps,
        "time_base": Rational(fps.denominator, fps.numerator),
        "sample_aspect_ratio": "1/1",
        "encoder_filter": "setsar=1",
        "mux_flags": ("fflags:+bitexact", "flags:+bitexact"),
        "color": color,
        "qualification": "per-segment-framehash-only",
    }
    return EncoderConfig(
        ffmpeg_sha256=ffmpeg_sha256,
        encoder="libx264",
        hardware_device=None,
        codec="h264",
        pixel_format="yuv420p",
        rate_control={"mode": "crf", "crf": crf},
        preset=preset,
        gop=gop,
        b_frames=0,
        output_fps=fps,
        output_time_base=Rational(fps.denominator, fps.numerator),
        color=color,
        self_test_sha256=canonical_sha256(contract, projection="identity"),
    )


def _candidate(
    candidates: tuple[MediaCandidate, ...],
    relative_path: str,
) -> MediaCandidate:
    matches = tuple(
        item for item in candidates if item.relative_path == relative_path
    )
    if len(matches) != 1:
        available = tuple(item.relative_path for item in candidates)
        raise ValueError(
            f"relative_video_path must match exactly one candidate; available={available}"
        )
    return matches[0]


def prepare_phase0(
    request: KagglePhase0Request,
    *,
    ffmpeg_path: Path,
    ffprobe_path: Path,
) -> PreparedPhase0:
    """Prepare immutable evidence and a resumable coordinator without running SR."""
    dataset = DatasetRoot(
        handle=request.dataset_handle,
        version=request.dataset_version,
        root=request.dataset_root,
        role="media",
    )
    candidate = _candidate(discover_media((dataset,)), request.relative_video_path)
    provisional_path = request.dataset_root / request.relative_video_path
    video_indexes = _video_indexes(provisional_path, ffprobe_path)
    selected_index = request.selected_video_index
    if selected_index is None:
        selected_index = video_indexes[0]
    if selected_index not in video_indexes:
        raise ValueError(
            f"selected_video_index must be one of {video_indexes}"
        )
    selection = select_media(
        candidate,
        dataset,
        video_stream_index=selected_index,
    )
    paths = create_job_paths(request.output_root, request.job_id)
    persist_selection(selection, paths.artifacts / "selection.json")
    media = probe_media(selection, ffprobe_path)
    video = _selected_video(media)
    color = classify_color(video)
    if color.state != "allowed" or color.output_color is None:
        raise ValueError(f"source color is not safely supported: {color.reason}")

    app = AppConfig(
        input_roots=(request.dataset_root.resolve(strict=True),),
        weight_roots=(request.dataset_root.resolve(strict=True),),
        wheel_roots=(request.dataset_root.resolve(strict=True),),
        output_root=request.output_root.resolve(strict=False),
    )
    if (
        media.duration_seconds is not None
        and media.duration_seconds > app.max_duration_seconds
    ):
        raise ValueError("source duration exceeds the configured 15-minute limit")
    try:
        job = JobConfig(
            selected_video_index=selected_index,
            target_width=request.target_width,
            target_height=request.target_height,
            fps_policy=request.fps_policy,
            target_fps=request.target_fps,
        ).validate_against(app)
    except ValueError as exc:
        raise ValueError(f"invalid job configuration: {exc}") from exc
    source_frames = tuple(stream_source_timing(media, ffprobe_path))
    preparation = plan_timing_preparation(video, source_frames)
    if preparation is not None:
        raise ValueError(
            "telecine source requires the full-span IVTC mezzanine stage before "
            "Phase 0; this notebook build refuses to chunk it directly"
        )
    timeline = plan_timeline(media, job, source_frames)
    geometry = plan_geometry(video, job, model_scale=2)
    from engvit.media.filters import build_timing_filter

    filter_graph = build_timing_filter(timeline, geometry, video)
    encoder = baseline_encoder_config(
        ffmpeg_path=ffmpeg_path,
        fps=timeline.output_fps,
        color=color.output_color,
        crf=request.crf,
        preset=request.preset,
    )
    environment = capture_environment(ffmpeg_path=ffmpeg_path)
    pre_identity = {
        "selection": selection,
        "timeline_sha256": timeline.sha256,
        "geometry": geometry,
        "encoder": encoder,
        "environment": environment,
        "pipeline": "phase0-lanczos-v1",
    }
    execution_identity = canonical_sha256(pre_identity, projection="identity")
    chunks = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=request.chunk_frames,
            max_scene_extension_frames=0,
            context_before_frames=2 if filter_graph.continuous else 0,
            context_after_frames=2 if filter_graph.continuous else 0,
            execution_identity_sha256=execution_identity,
        ),
    )
    coordinator = Coordinator(
        manifest_path=paths.artifacts / "manifest.json",
        segments_root=paths.segments,
        chunks=chunks,
        job_identity_sha256=execution_identity,
    )
    writer = AtomicArtifactWriter()
    writer.write(
        paths.artifacts / "timeline.json",
        canonical_bytes(timeline, projection="full"),
    )
    writer.write(
        paths.artifacts / "geometry.json",
        canonical_bytes(geometry, projection="full"),
    )
    writer.write(
        paths.artifacts / "environment.json",
        canonical_bytes(environment, projection="full"),
    )
    return PreparedPhase0(
        request=request,
        paths=paths,
        selection=selection,
        media=media,
        video=video,
        color=color,
        timeline=timeline,
        geometry=geometry,
        filter_graph=filter_graph,
        encoder=encoder,
        chunks=chunks,
        environment=environment,
        execution_identity_sha256=execution_identity,
        coordinator=coordinator,
        ffmpeg_path=ffmpeg_path,
    )


def run_prepared_phase0(
    prepared: PreparedPhase0,
    *,
    max_new_chunks: int | None = None,
    resume_paused: bool = False,
) -> Phase0Result:
    """Execute or resume a prepared Phase 0 job."""
    return execute_lanczos_job(
        source=prepared.selection.canonical_path,
        video_stream_index=prepared.selection.video_stream_index,
        timeline=prepared.timeline,
        filter_graph=prepared.filter_graph,
        chunks=prepared.chunks,
        encoder=prepared.encoder,
        output_size=prepared.geometry.target_size,
        ffmpeg_path=prepared.ffmpeg_path,
        coordinator=prepared.coordinator,
        output_path=prepared.paths.artifacts / "enhanced-video.mkv",
        worker_id="kaggle-worker-0",
        max_new_chunks=max_new_chunks,
        resume_paused=resume_paused,
    )
