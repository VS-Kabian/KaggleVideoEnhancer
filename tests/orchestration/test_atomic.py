from __future__ import annotations

from pathlib import Path

import pytest

from engvit.orchestration.atomic import AtomicArtifactWriter


def test_failure_after_fsync_before_replace_keeps_old_destination(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"old")

    def fail(stage: str) -> None:
        if stage == "after_file_fsync":
            raise RuntimeError("simulated kill")

    writer = AtomicArtifactWriter(stage_hook=fail)
    with pytest.raises(RuntimeError, match="simulated kill"):
        writer.write(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert not tuple(tmp_path.glob("*.partial"))


def test_failure_after_replace_is_recoverable_from_complete_new_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"

    def fail(stage: str) -> None:
        if stage == "after_replace":
            raise RuntimeError("simulated kill")

    writer = AtomicArtifactWriter(stage_hook=fail)
    with pytest.raises(RuntimeError, match="simulated kill"):
        writer.write(destination, b"complete-new")
    assert destination.read_bytes() == b"complete-new"


def test_compare_and_replace_rejects_lost_update(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    writer = AtomicArtifactWriter()
    first = writer.write(destination, b"generation-1")
    destination.write_bytes(b"someone-else")
    with pytest.raises(ValueError, match="changed"):
        writer.write(
            destination,
            b"generation-2",
            expected_previous_sha256=first.sha256,
        )

