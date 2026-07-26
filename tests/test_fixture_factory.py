from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fixture_manifest_is_complete_and_root_independent(tmp_path: Path) -> None:
    """Catches missing adversarial media classes or host paths in fixture identity."""
    script = Path("scripts/generate_media_fixtures.py").resolve()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(first_root),
            "--manifest-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(second_root),
            "--manifest-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_bytes = (first_root / "fixtures.manifest.json").read_bytes()
    second_bytes = (second_root / "fixtures.manifest.json").read_bytes()
    assert first_bytes == second_bytes

    manifest = json.loads(first_bytes)
    assert {item["fixture_id"] for item in manifest["fixtures"]} == {
        "b_frames",
        "bt601_limited",
        "bt709_full",
        "cfr_30000_1001",
        "hlg",
        "interlaced",
        "irregular_vfr",
        "malformed_probe",
        "multi_stream",
        "negative_start_pts",
        "non_square_sar",
        "pq",
        "rotation_flip",
        "telecine_23",
    }
    assert manifest["schema_version"] == "1"
    assert manifest["ffmpeg_contract"] == "imageio-ffmpeg-0.6.0/ffmpeg-7.1"
    assert str(tmp_path).replace("\\", "/") not in first_bytes.decode("utf-8")
