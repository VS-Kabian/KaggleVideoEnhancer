from __future__ import annotations

import hashlib
from pathlib import Path

from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from engvit.types import ChunkCompletion, ChunkSpec
from tests.analysis.helpers import timeline


def chunks(identity: str = "a" * 64) -> tuple[ChunkSpec, ...]:
    return plan_chunks(
        timeline(12),
        (),
        ChunkPolicy(
            target_core_frames=4,
            max_scene_extension_frames=0,
            context_before_frames=1,
            context_after_frames=1,
            execution_identity_sha256=identity,
        ),
    )


def completion(
    chunk: ChunkSpec,
    lease_id: str,
    segments: Path,
    *,
    payload: bytes = b"valid segment",
) -> ChunkCompletion:
    path = segments / f"{chunk.chunk_id}.mkv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ChunkCompletion(
        chunk_id=chunk.chunk_id,
        lease_id=lease_id,
        identity_sha256=chunk.identity_sha256,
        partial_path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        frame_count=chunk.output_core_end - chunk.output_core_start,
        first_pts=chunk.output_core_start,
        last_pts=chunk.output_core_end - 1,
        boundary_frame_hashes=("b" * 64, "c" * 64),
        encoder_extradata_sha256="d" * 64,
        observations={},
    )

