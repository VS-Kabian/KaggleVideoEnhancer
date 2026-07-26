from __future__ import annotations

from pathlib import Path

from engvit.smoke import run_phase0_smoke
from tests.media.pipeline_helpers import ffmpeg_path


def test_smoke_artifact_is_byte_identical_across_clean_roots(
    tmp_path: Path,
) -> None:
    first = run_phase0_smoke(tmp_path / "first", ffmpeg_path())
    second = run_phase0_smoke(tmp_path / "second", ffmpeg_path())

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.artifact.path.read_bytes() == second.artifact.path.read_bytes()
