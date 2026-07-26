from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engvit.supply_chain import (
    require_safe_runtime_weight,
    validate_hash_locked_requirements,
    verify_wheelhouse,
)


def write_lock(root: Path, files: list[dict[str, object]]) -> Path:
    lock = root / "wheelhouse-lock.json"
    lock.write_text(
        json.dumps({"schema_version": "1", "files": files}),
        encoding="utf-8",
    )
    return lock


def test_verify_wheelhouse_accepts_exact_binary_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "safe_pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"audited-wheel")
    lock = write_lock(
        tmp_path,
        [
            {
                "filename": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "bytes": wheel.stat().st_size,
                "source": "https://files.pythonhosted.org/example",
            }
        ],
    )
    report = verify_wheelhouse(tmp_path, lock)
    assert report.passed is True
    assert report.failures == ()


def test_verify_wheelhouse_rejects_hash_mismatch(tmp_path: Path) -> None:
    """Catches a wheel modified after review."""
    wheel = tmp_path / "safe_pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"modified")
    lock = write_lock(
        tmp_path,
        [
            {
                "filename": wheel.name,
                "sha256": "0" * 64,
                "bytes": wheel.stat().st_size,
                "source": "https://files.pythonhosted.org/example",
            }
        ],
    )
    report = verify_wheelhouse(tmp_path, lock)
    assert report.passed is False
    assert f"sha256_mismatch:{wheel.name}" in report.failures


def test_verify_wheelhouse_rejects_sdist_and_vcs_source(tmp_path: Path) -> None:
    """Catches executable source builds entering the offline wheelhouse."""
    archive = tmp_path / "unsafe-1.0.tar.gz"
    archive.write_bytes(b"source")
    lock = write_lock(
        tmp_path,
        [
            {
                "filename": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
                "source": "git+https://example.invalid/repo.git",
            }
        ],
    )
    report = verify_wheelhouse(tmp_path, lock)
    assert report.passed is False
    assert f"not_binary_wheel:{archive.name}" in report.failures
    assert f"vcs_source:{archive.name}" in report.failures


def test_runtime_weight_accepts_only_contained_safetensors(tmp_path: Path) -> None:
    """Catches a pickle checkpoint or escaped path reaching model loading."""
    root = tmp_path / "weights"
    nested = root / "approved"
    nested.mkdir(parents=True)
    safe = nested / "model.safetensors"
    legacy = nested / "model.pth"
    outside = tmp_path / "outside.safetensors"
    safe.write_bytes(b"safe")
    legacy.write_bytes(b"pickle")
    outside.write_bytes(b"outside")

    assert require_safe_runtime_weight(safe, root) == safe.resolve()
    with pytest.raises(ValueError, match="safetensors"):
        require_safe_runtime_weight(legacy, root)
    with pytest.raises(ValueError, match="contained"):
        require_safe_runtime_weight(outside, root)


def test_requirement_lock_requires_exact_versions_and_hashes() -> None:
    valid = "pydantic==2.13.4 --hash=sha256:" + ("a" * 64)
    assert validate_hash_locked_requirements(valid) == ()

    failures = validate_hash_locked_requirements(
        "\n".join(
            (
                "numpy>=2",
                "package==1.0",
                "git+https://example.invalid/repo.git@deadbeef",
            )
        )
    )
    assert failures == (
        "line_1:not_exactly_pinned",
        "line_1:missing_sha256",
        "line_2:missing_sha256",
        "line_3:vcs_or_url",
        "line_3:not_exactly_pinned",
        "line_3:missing_sha256",
    )
