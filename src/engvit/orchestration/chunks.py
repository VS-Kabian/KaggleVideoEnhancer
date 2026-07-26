"""Scene-aware chunk cores and source/context coordinate planning."""

from __future__ import annotations

from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from engvit.analysis.scenes import Scene
from engvit.canonical import canonical_sha256
from engvit.types import ChunkSpec, TimelinePlan


class ChunkPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_core_frames: int = Field(ge=1)
    max_scene_extension_frames: int = Field(ge=0)
    context_before_frames: int = Field(ge=0)
    context_after_frames: int = Field(ge=0)
    execution_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _boundary(
    start: int,
    frame_count: int,
    scene_ends: tuple[int, ...],
    policy: ChunkPolicy,
) -> int:
    target = min(frame_count, start + policy.target_core_frames)
    if target == frame_count:
        return frame_count
    candidates = tuple(
        boundary
        for boundary in scene_ends
        if boundary > start
        and abs(boundary - target) <= policy.max_scene_extension_frames
    )
    if not candidates:
        return target
    return min(candidates, key=lambda value: (abs(value - target), value))


def _source_range(
    timeline: TimelinePlan,
    start: int,
    end: int,
) -> tuple[int, int]:
    source_indexes = {
        source_index
        for output in timeline.output_frames[start:end]
        for source_index in output.source_indexes
    }
    if not source_indexes:
        raise ValueError("chunk has no source-frame mapping")
    source_frames = tuple(timeline.source_frames[index] for index in source_indexes)
    decode_start = min(frame.best_effort_pts for frame in source_frames)
    decode_end = max(
        frame.best_effort_pts + frame.duration_pts for frame in source_frames
    )
    return decode_start, decode_end


def plan_chunks(
    timeline: TimelinePlan,
    scenes: tuple[Scene, ...],
    policy: ChunkPolicy,
) -> tuple[ChunkSpec, ...]:
    """Cover normalized output exactly once with explicit decode context."""
    frame_count = len(timeline.output_frames)
    if frame_count == 0:
        raise ValueError("cannot chunk an empty timeline")
    if scenes:
        if scenes[0].start_frame != 0 or scenes[-1].end_frame != frame_count:
            raise ValueError("scene coordinates must cover the output timeline")
        for left, right in pairwise(scenes):
            if left.end_frame != right.start_frame:
                raise ValueError("scene coordinates must be contiguous")
    scene_ends = tuple(scene.end_frame for scene in scenes[:-1])
    chunks: list[ChunkSpec] = []
    start = 0
    while start < frame_count:
        end = _boundary(start, frame_count, scene_ends, policy)
        if end <= start:
            raise ValueError("chunk policy did not advance the output core")
        expanded_start = max(0, start - policy.context_before_frames)
        expanded_end = min(frame_count, end + policy.context_after_frames)
        decode_start, decode_end = _source_range(
            timeline,
            expanded_start,
            expanded_end,
        )
        scene_ids = tuple(
            scene.scene_id
            for scene in scenes
            if scene.start_frame < end and scene.end_frame > start
        )
        identity_payload = {
            "execution_identity_sha256": policy.execution_identity_sha256,
            "timeline_sha256": timeline.sha256,
            "source_decode_start_pts": decode_start,
            "source_decode_end_pts": decode_end,
            "output_core_start": start,
            "output_core_end": end,
            "context_before": start - expanded_start,
            "context_after": expanded_end - end,
            "scene_ids": scene_ids,
        }
        chunk_id = f"chunk-{len(chunks):06d}"
        chunks.append(
            ChunkSpec(
                chunk_id=chunk_id,
                source_decode_start_pts=decode_start,
                source_decode_end_pts=decode_end,
                output_core_start=start,
                output_core_end=end,
                context_before=start - expanded_start,
                context_after=expanded_end - end,
                scene_ids=scene_ids,
                identity_sha256=canonical_sha256(
                    identity_payload,
                    projection="identity",
                ),
            )
        )
        start = end
    return tuple(chunks)
