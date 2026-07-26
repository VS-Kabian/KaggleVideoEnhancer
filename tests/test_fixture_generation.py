from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")


@pytest.mark.integration
def test_pinned_ffmpeg_generates_byte_identical_fixtures(tmp_path: Path) -> None:
    """Catches volatile container metadata or host paths changing fixture bytes."""
    script = Path("scripts/generate_media_fixtures.py").resolve()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    roots = (tmp_path / "fixture-root-a", tmp_path / "fixture-root-b")

    manifests: list[dict[str, object]] = []
    for root in roots:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                str(root),
                "--ffmpeg",
                ffmpeg,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        manifests.append(json.loads((root / "fixtures.hashes.json").read_bytes()))

    first_hashes = {
        item["fixture_id"]: item["sha256"]  # type: ignore[index]
        for item in manifests[0]["files"]  # type: ignore[union-attr]
    }
    second_hashes = {
        item["fixture_id"]: item["sha256"]  # type: ignore[index]
        for item in manifests[1]["files"]  # type: ignore[union-attr]
    }
    assert first_hashes == second_hashes

