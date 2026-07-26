"""Generate EngVit's small, deterministic adversarial media fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from engvit.testing.fixtures import FixtureSpec, fixture_specs

FFMPEG_CONTRACT = "imageio-ffmpeg-0.6.0/ffmpeg-7.1"
_GLOBAL_ARGS = (
    "-hide_banner",
    "-loglevel",
    "error",
    "-nostdin",
    "-y",
)
_OUTPUT_ARGS = (
    "-bitexact",
    "-flags",
    "+bitexact",
    "-map_metadata",
    "-1",
    "-threads",
    "1",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "ffmpeg_contract": FFMPEG_CONTRACT,
        "fixtures": [
            {
                "fixture_id": spec.fixture_id,
                "output_file": spec.output_file,
                "args": list(spec.args),
                "purpose": spec.purpose,
            }
            for spec in fixture_specs()
        ],
    }


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(payload)
    partial.replace(path)


def _support_files(output: Path) -> dict[str, Path]:
    subtitle = output / "fixture.srt"
    chapters = output / "fixture.ffmetadata"
    attachment = output / "fixture-attachment.txt"
    _write_atomic(
        subtitle,
        b"1\n00:00:00,250 --> 00:00:01,750\nEngVit forced subtitle\n",
    )
    _write_atomic(
        chapters,
        b";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=One\n",
    )
    _write_atomic(attachment, b"EngVit attachment canary\n")
    return {
        "{subtitle}": subtitle,
        "{chapters}": chapters,
        "{attachment}": attachment,
    }


def _materialize_args(
    spec: FixtureSpec,
    output: Path,
    support: dict[str, Path],
) -> list[str]:
    output_path = output / spec.output_file
    replacements = {**support, "{output}": output_path}
    raw_args = list(spec.args)
    output_index = raw_args.index("{output}")
    raw_args[output_index:output_index] = _OUTPUT_ARGS
    return [
        str(replacements.get(argument, argument))
        for argument in (*_GLOBAL_ARGS, *raw_args)
    ]


def _ffmpeg_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "FFmpeg version probe failed")
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if "ffmpeg version 7.1" not in first_line:
        raise RuntimeError(
            f"fixture generation requires {FFMPEG_CONTRACT}; observed {first_line!r}"
        )
    return first_line


def _generate(output: Path, ffmpeg: str) -> dict[str, Any]:
    support = _support_files(output)
    records: list[dict[str, Any]] = []
    for spec in fixture_specs():
        destination = output / spec.output_file
        if spec.fixture_id == "malformed_probe":
            _write_atomic(
                destination,
                _canonical_bytes(
                    {
                        "format": {"duration": "not-a-decimal"},
                        "streams": [
                            {
                                "index": 0,
                                "codec_type": "video",
                                "avg_frame_rate": "0/0",
                                "sample_aspect_ratio": "broken",
                            }
                        ],
                    }
                ),
            )
        else:
            command = [ffmpeg, *_materialize_args(spec, output, support)]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{spec.fixture_id} generation failed: {completed.stderr.strip()}"
                )
        payload = destination.read_bytes()
        records.append(
            {
                "fixture_id": spec.fixture_id,
                "output_file": spec.output_file,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {"ffmpeg_version": _ffmpeg_version(ffmpeg), "files": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    _write_atomic(output / "fixtures.manifest.json", _canonical_bytes(manifest))
    if args.manifest_only:
        return 0

    executable = shutil.which(args.ffmpeg) if not Path(args.ffmpeg).is_file() else args.ffmpeg
    if not executable:
        print(
            f"FFmpeg not found: {args.ffmpeg}. Generate in the pinned Kaggle/test image.",
            file=sys.stderr,
        )
        return 2
    try:
        hashes = _generate(output, str(executable))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    _write_atomic(output / "fixtures.hashes.json", _canonical_bytes(hashes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
