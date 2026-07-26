"""Worker-side immutable completion validation helpers."""

from __future__ import annotations

from engvit.types import ChunkCompletion, ChunkSpec


def validate_worker_completion(
    chunk: ChunkSpec,
    completion: ChunkCompletion,
) -> ChunkCompletion:
    """Validate identity and coordinates before queueing a worker result."""
    if completion.chunk_id != chunk.chunk_id:
        raise ValueError("worker completion references another chunk")
    if completion.identity_sha256 != chunk.identity_sha256:
        raise ValueError("worker completion identity does not match")
    if completion.frame_count != chunk.output_core_end - chunk.output_core_start:
        raise ValueError("worker completion frame count does not match")
    if (
        completion.first_pts != chunk.output_core_start
        or completion.last_pts != chunk.output_core_end - 1
    ):
        raise ValueError("worker completion PTS does not match")
    return completion

