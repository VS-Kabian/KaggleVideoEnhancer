from __future__ import annotations

from pathlib import Path

from engvit.orchestration.coordinator import Coordinator
from tests.orchestration.helpers import chunks, completion


def test_recover_accepts_valid_sidecar_after_rename_before_manifest_commit(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    segments = tmp_path / "segments"
    service = Coordinator(
        manifest_path=manifest_path,
        segments_root=segments,
        chunks=chunks(),
        job_identity_sha256="e" * 64,
    )
    lease = service.lease("worker")
    assert lease is not None
    candidate = completion(lease.chunk, lease.lease_id, segments)
    service.stage_completion(candidate)

    restarted = Coordinator(
        manifest_path=manifest_path,
        segments_root=segments,
        chunks=chunks(),
        job_identity_sha256="e" * 64,
    )
    report = restarted.recover()
    assert lease.chunk.chunk_id in report.repaired_chunks
    assert restarted.current.chunks[0].status == "complete"


def test_recover_rejects_corrupt_complete_segment(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    segments = tmp_path / "segments"
    service = Coordinator(
        manifest_path=manifest_path,
        segments_root=segments,
        chunks=chunks(),
        job_identity_sha256="e" * 64,
    )
    lease = service.lease("worker")
    assert lease is not None
    candidate = completion(lease.chunk, lease.lease_id, segments)
    service.commit(candidate)
    candidate.partial_path.write_bytes(b"corrupt but present")
    report = service.recover()
    assert lease.chunk.chunk_id in report.reset_chunks
    assert service.current.chunks[0].status == "pending"
