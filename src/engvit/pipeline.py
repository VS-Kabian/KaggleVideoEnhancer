"""Runnable deterministic Phase 0 Lanczos job."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from engvit.media.concat import VideoArtifact, validate_and_concat
from engvit.media.decode import decode_normalized_frames
from engvit.media.encode import encode_segment
from engvit.media.filters import FilterGraph
from engvit.orchestration.coordinator import Coordinator
from engvit.orchestration.manifest import CompletionRecord
from engvit.tiling.runner import enhance_tiled
from engvit.types import (
    ChunkCompletion,
    ChunkSpec,
    EncoderConfig,
    FrameEnhancer,
    JSONValue,
    TilePolicy,
    TimelinePlan,
)


@dataclass(frozen=True)
class Phase0Result:
    state: str
    artifact: VideoArtifact | None
    completed_chunks: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion(record: CompletionRecord) -> ChunkCompletion:
    observations = cast(dict[str, JSONValue], dict(record.observations))
    return ChunkCompletion(
        chunk_id=record.chunk_id,
        lease_id=record.lease_id,
        identity_sha256=record.identity_sha256,
        partial_path=record.partial_path,
        bytes=record.bytes,
        sha256=record.sha256,
        frame_count=record.frame_count,
        first_pts=record.first_pts,
        last_pts=record.last_pts,
        boundary_frame_hashes=record.boundary_frame_hashes,
        encoder_extradata_sha256=record.encoder_extradata_sha256,
        observations=observations,
    )


def execute_lanczos_job(
    *,
    source: Path,
    video_stream_index: int,
    timeline: TimelinePlan,
    filter_graph: FilterGraph,
    chunks: tuple[ChunkSpec, ...],
    encoder: EncoderConfig,
    output_size: tuple[int, int],
    ffmpeg_path: Path,
    coordinator: Coordinator,
    output_path: Path,
    worker_id: str,
    max_new_chunks: int | None = None,
    resume_paused: bool = False,
) -> Phase0Result:
    """Recover and execute verified chunks until complete or explicitly paused."""
    return _execute_spatial_job(
        source=source,
        video_stream_index=video_stream_index,
        timeline=timeline,
        filter_graph=filter_graph,
        chunks=chunks,
        encoder=encoder,
        decode_size=output_size,
        output_size=output_size,
        ffmpeg_path=ffmpeg_path,
        coordinator=coordinator,
        output_path=output_path,
        worker_id=worker_id,
        frame_transform=None,
        max_new_chunks=max_new_chunks,
        resume_paused=resume_paused,
    )


def _execute_spatial_job(
    *,
    source: Path,
    video_stream_index: int,
    timeline: TimelinePlan,
    filter_graph: FilterGraph,
    chunks: tuple[ChunkSpec, ...],
    encoder: EncoderConfig,
    decode_size: tuple[int, int],
    output_size: tuple[int, int],
    ffmpeg_path: Path,
    coordinator: Coordinator,
    output_path: Path,
    worker_id: str,
    frame_transform: Callable[[NDArray[np.uint8]], NDArray[np.uint8]] | None,
    max_new_chunks: int | None,
    resume_paused: bool,
) -> Phase0Result:
    """Shared verified spatial worker used by Lanczos and approved neural paths."""
    if _sha256(ffmpeg_path) != encoder.ffmpeg_sha256:
        raise ValueError("active FFmpeg binary does not match the encoder plan")
    if encoder.output_fps != timeline.output_fps:
        raise ValueError("encoder FPS does not match the timeline")
    if encoder.output_time_base != timeline.output_time_base:
        raise ValueError("encoder time base does not match the timeline")
    expected_specs = {chunk.chunk_id: chunk.identity_sha256 for chunk in chunks}
    if expected_specs != coordinator.chunk_identities:
        raise ValueError("coordinator chunks do not match the execution request")

    if max_new_chunks is not None and max_new_chunks < 1:
        raise ValueError("max_new_chunks must be positive when provided")
    coordinator.recover()
    if coordinator.current.state == "paused" and resume_paused:
        coordinator.resume()
    chunks_processed = 0
    while True:
        lease = coordinator.lease(worker_id)
        if lease is None:
            break
        frames = decode_normalized_frames(
            source=source,
            video_stream_index=video_stream_index,
            chunk=lease.chunk,
            plan=timeline,
            filter_graph=filter_graph,
            output_size=decode_size,
            ffmpeg_path=ffmpeg_path,
        )
        encoded_frames = (
            frames
            if frame_transform is None
            else (frame_transform(frame) for frame in frames)
        )
        completion = encode_segment(
            frames=encoded_frames,
            chunk=lease.chunk,
            lease_id=lease.lease_id,
            config=encoder,
            frame_size=output_size,
            output_path=coordinator.segments_root / f"{lease.chunk.chunk_id}.mkv",
            ffmpeg_path=ffmpeg_path,
        )
        coordinator.commit(completion)
        chunks_processed += 1
        if (
            max_new_chunks is not None
            and chunks_processed >= max_new_chunks
            and coordinator.current.state != "complete"
        ):
            coordinator.request_pause()
            break

    complete_records = tuple(
        record.completion
        for record in coordinator.current.chunks
        if record.completion is not None
    )
    completed_ids = tuple(record.chunk_id for record in complete_records)
    if coordinator.current.state != "complete":
        return Phase0Result(
            state=coordinator.current.state,
            artifact=None,
            completed_chunks=completed_ids,
        )
    if len(complete_records) != len(chunks):
        raise ValueError("complete manifest does not contain every chunk completion")
    artifact = validate_and_concat(
        tuple(_completion(record) for record in complete_records),
        encoder,
        output_path=output_path,
        ffmpeg_path=ffmpeg_path,
    )
    return Phase0Result(
        state="complete",
        artifact=artifact,
        completed_chunks=completed_ids,
    )


def execute_neural_job(
    *,
    source: Path,
    video_stream_index: int,
    timeline: TimelinePlan,
    filter_graph: FilterGraph,
    chunks: tuple[ChunkSpec, ...],
    encoder: EncoderConfig,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    enhancer: FrameEnhancer,
    tile_policy: TilePolicy,
    ffmpeg_path: Path,
    coordinator: Coordinator,
    output_path: Path,
    worker_id: str,
    max_new_chunks: int | None = None,
    resume_paused: bool = False,
) -> Phase0Result:
    """Run an already approved enhancer; registry/admission gates occur upstream."""
    expected = (
        input_size[0] * enhancer.scale,
        input_size[1] * enhancer.scale,
    )
    if output_size != expected:
        raise ValueError(
            "neural worker currently requires native model output geometry; "
            "a separately qualified final resize is unavailable"
        )

    def transform(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return enhance_tiled(frame, enhancer, tile_policy)

    return _execute_spatial_job(
        source=source,
        video_stream_index=video_stream_index,
        timeline=timeline,
        filter_graph=filter_graph,
        chunks=chunks,
        encoder=encoder,
        decode_size=input_size,
        output_size=output_size,
        ffmpeg_path=ffmpeg_path,
        coordinator=coordinator,
        output_path=output_path,
        worker_id=worker_id,
        frame_transform=transform,
        max_new_chunks=max_new_chunks,
        resume_paused=resume_paused,
    )
