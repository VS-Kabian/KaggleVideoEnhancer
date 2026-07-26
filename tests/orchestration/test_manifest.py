from __future__ import annotations

from pathlib import Path

from engvit.orchestration.coordinator import Coordinator
from engvit.orchestration.manifest import Manifest
from tests.orchestration.helpers import chunks, completion


def test_manifest_round_trip_preserves_generation_and_completion(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    service = Coordinator(
        manifest_path=manifest_path,
        segments_root=tmp_path / "segments",
        chunks=chunks(),
        job_identity_sha256="e" * 64,
    )
    lease = service.lease("worker")
    assert lease is not None
    service.commit(completion(lease.chunk, lease.lease_id, tmp_path / "segments"))
    loaded = Manifest.model_validate_json(manifest_path.read_bytes())
    assert loaded.generation == 2
    assert loaded.chunks[0].status == "complete"
    assert loaded.chunks[0].completion is not None

