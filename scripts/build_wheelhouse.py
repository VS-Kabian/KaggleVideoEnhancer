"""Build a binary-only wheelhouse from a fully hash-locked requirements file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from engvit.supply_chain import validate_hash_locked_requirements


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(payload)
    partial.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    requirements = args.requirements_lock.resolve(strict=True)
    failures = validate_hash_locked_requirements(
        requirements.read_text(encoding="utf-8")
    )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2

    output = args.output.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "--dest",
        str(output),
        "--requirement",
        str(requirements),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    wheels = sorted(output.glob("*.whl"))
    entries = [
        {
            "filename": wheel.name,
            "sha256": _sha256(wheel),
            "bytes": wheel.stat().st_size,
            "source": f"hash-locked:{requirements.name}",
        }
        for wheel in wheels
    ]
    _write_atomic(
        output / "wheelhouse-lock.json",
        {"schema_version": "1", "files": entries},
    )
    _write_atomic(
        output / "wheelhouse-sbom.json",
        {
            "schema_version": "1",
            "requirements_sha256": _sha256(requirements),
            "artifacts": entries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
