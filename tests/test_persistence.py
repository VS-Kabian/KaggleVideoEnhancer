from __future__ import annotations

from pathlib import Path

import pytest

from engvit.paths import create_job_paths
from engvit.persistence import prepare_continuation, resume_continuation


def test_continuation_round_trip_verifies_every_file(tmp_path: Path) -> None:
    job = create_job_paths(tmp_path / "jobs", "source")
    (job.artifacts / "manifest.json").write_bytes(b"manifest")
    (job.artifacts / "timeline.json").write_bytes(b"timeline")
    (job.segments / "chunk-000000.mkv").write_bytes(b"segment")
    receipt = prepare_continuation(
        job,
        tmp_path / "private-continuation.zip",
    )
    resumed = resume_continuation(
        receipt,
        attached_root=tmp_path,
        output_root=tmp_path / "resumed",
        job_id="restored",
    )
    assert (resumed.artifacts / "manifest.json").read_bytes() == b"manifest"
    assert (resumed.segments / "chunk-000000.mkv").read_bytes() == b"segment"


def test_tampered_continuation_is_rejected(tmp_path: Path) -> None:
    job = create_job_paths(tmp_path / "jobs", "source")
    (job.artifacts / "manifest.json").write_bytes(b"manifest")
    receipt = prepare_continuation(job, tmp_path / "continuation.zip")
    receipt.archive_path.write_bytes(receipt.archive_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match=r"hash|size"):
        resume_continuation(
            receipt,
            attached_root=tmp_path,
            output_root=tmp_path / "resumed",
            job_id="restored",
        )
