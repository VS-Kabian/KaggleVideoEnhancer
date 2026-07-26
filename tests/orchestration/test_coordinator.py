from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from engvit.orchestration.coordinator import Coordinator
from tests.orchestration.helpers import chunks, completion


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def coordinator(tmp_path: Path, clock: Clock | None = None) -> Coordinator:
    if clock is not None:
        return Coordinator(
            manifest_path=tmp_path / "manifest.json",
            segments_root=tmp_path / "segments",
            chunks=chunks(),
            job_identity_sha256="e" * 64,
            lease_seconds=10,
            clock=clock,
        )
    return Coordinator(
        manifest_path=tmp_path / "manifest.json",
        segments_root=tmp_path / "segments",
        chunks=chunks(),
        job_identity_sha256="e" * 64,
        lease_seconds=10,
    )


def test_simultaneous_workers_receive_unique_chunks(tmp_path: Path) -> None:
    service = coordinator(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        leases = tuple(pool.map(service.lease, ("one", "two", "three")))
    assert all(lease is not None for lease in leases)
    assert len({lease.chunk.chunk_id for lease in leases if lease is not None}) == 3
    assert service.current.generation == 3


def test_stale_lease_returns_to_pending_and_can_be_released(tmp_path: Path) -> None:
    clock = Clock()
    service = coordinator(tmp_path, clock)
    first = service.lease("lost-worker")
    assert first is not None
    clock.value = 111.0
    second = service.lease("replacement")
    assert second is not None
    assert second.chunk.chunk_id == first.chunk.chunk_id
    assert second.lease_id != first.lease_id


def test_commit_validates_hash_frames_pts_boundaries_and_identity(
    tmp_path: Path,
) -> None:
    service = coordinator(tmp_path)
    lease = service.lease("worker")
    assert lease is not None
    candidate = completion(lease.chunk, lease.lease_id, tmp_path / "segments")
    candidate.partial_path.write_bytes(b"wrong same-ish")
    with pytest.raises(ValueError, match=r"hash|size"):
        service.commit(candidate)
    assert service.current.chunks[0].status == "leased"


def test_pause_transitions_only_after_committed_segment(tmp_path: Path) -> None:
    service = coordinator(tmp_path)
    lease = service.lease("worker")
    assert lease is not None
    service.request_pause()
    assert service.current.state == "pause_requested"
    assert service.lease("other") is None
    generation = service.commit(
        completion(lease.chunk, lease.lease_id, tmp_path / "segments")
    )
    assert generation.state == "paused"
    assert service.current.state == "paused"


def test_stale_worker_cannot_commit_after_chunk_is_released(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service = coordinator(tmp_path, clock)
    stale = service.lease("lost")
    assert stale is not None
    clock.value = 111.0
    replacement = service.lease("replacement")
    assert replacement is not None
    candidate = completion(stale.chunk, stale.lease_id, tmp_path / "segments")
    with pytest.raises(ValueError, match="lease"):
        service.commit(candidate)
