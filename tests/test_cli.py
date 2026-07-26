from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import imageio_ffmpeg

from engvit.cli import main


def _invoke(arguments: list[str]) -> tuple[int, dict[str, object], str]:
    stdout = StringIO()
    stderr = StringIO()
    status = main(arguments, stdout=stdout, stderr=stderr)
    payload = json.loads(stdout.getvalue()) if stdout.getvalue() else {}
    return status, payload, stderr.getvalue()


def test_preflight_returns_nonzero_for_public_input(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "notebook_id": "owner/private-upscale",
                "notebook_visibility": "private",
                "internet_enabled": False,
                "datasets": [
                    {
                        "handle": "owner/media",
                        "version": "7",
                        "role": "media",
                        "visibility": "public",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status, payload, error = _invoke(["preflight", "--context", str(context)])

    assert status == 2
    assert payload["passed"] is False
    assert payload["failures"] == ["dataset_public:owner/media"]
    assert error == ""


def test_discover_never_emits_absolute_source_paths(tmp_path: Path) -> None:
    media_root = tmp_path / "private-media"
    media_root.mkdir()
    (media_root / "clip.mp4").write_bytes(b"not-decoded-during-discovery")
    (media_root / "notes.txt").write_text("ignore", encoding="utf-8")

    status, payload, error = _invoke(
        [
            "discover",
            "--dataset-root",
            str(media_root),
            "--handle",
            "owner/media",
            "--version",
            "7",
        ]
    )

    assert status == 0
    assert payload["count"] == 1
    assert payload["candidates"] == [
        {
            "bytes": 28,
            "dataset_handle": "owner/media",
            "dataset_version": "7",
            "relative_path": "clip.mp4",
        }
    ]
    assert str(media_root) not in json.dumps(payload)
    assert error == ""


def test_invalid_command_failure_is_redacted(tmp_path: Path) -> None:
    missing = tmp_path / "secret-name.json"

    status, payload, error = _invoke(
        ["preflight", "--context", str(missing)]
    )

    assert status == 2
    assert payload == {}
    assert "secret-name" not in error
    assert "FileNotFoundError" in error


def test_benchmark_command_labels_smoke_as_non_admitting(
    tmp_path: Path,
) -> None:
    status, payload, error = _invoke(
        [
            "benchmark",
            "--output-root",
            str(tmp_path),
            "--ffmpeg",
            imageio_ffmpeg.get_ffmpeg_exe(),
        ]
    )

    assert status == 0
    assert payload["state"] == "complete"
    assert payload["scope"] == "six_frame_plumbing_smoke_only"
    assert payload["admissible_for_full_job"] is False
    assert error == ""
