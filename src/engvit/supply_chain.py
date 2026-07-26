"""Verification of offline, binary-only, hash-locked dependencies."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class WheelLockEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    source: str = Field(min_length=1)


class WheelhouseLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    files: tuple[WheelLockEntry, ...]


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    failures: tuple[str, ...]
    verified_files: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheelhouse(root: Path, lock_path: Path) -> VerificationReport:
    """Verify an attached wheelhouse without importing or executing its contents."""
    failures: list[str] = []
    verified: list[str] = []
    try:
        lock = WheelhouseLock.model_validate_json(lock_path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        return VerificationReport(
            passed=False,
            failures=(f"invalid_lock:{type(exc).__name__}",),
            verified_files=(),
        )

    approved_root = root.resolve(strict=True)
    seen: set[str] = set()
    for entry in lock.files:
        if entry.filename in seen:
            failures.append(f"duplicate_lock_entry:{entry.filename}")
            continue
        seen.add(entry.filename)
        if Path(entry.filename).name != entry.filename:
            failures.append(f"unsafe_filename:{entry.filename}")
            continue
        if not entry.filename.lower().endswith(".whl"):
            failures.append(f"not_binary_wheel:{entry.filename}")
        lowered_source = entry.source.lower()
        if lowered_source.startswith(("git+", "hg+", "svn+", "bzr+")):
            failures.append(f"vcs_source:{entry.filename}")

        candidate = approved_root / entry.filename
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            failures.append(f"missing_file:{entry.filename}")
            continue
        if resolved.parent != approved_root or resolved.is_symlink() or not resolved.is_file():
            failures.append(f"unsafe_file:{entry.filename}")
            continue
        if resolved.stat().st_size != entry.bytes:
            failures.append(f"size_mismatch:{entry.filename}")
            continue
        if _sha256(resolved) != entry.sha256:
            failures.append(f"sha256_mismatch:{entry.filename}")
            continue
        verified.append(entry.filename)

    unlisted_wheels = {
        path.name
        for path in approved_root.glob("*.whl")
        if path.is_file() and path.name not in seen
    }
    failures.extend(f"unlisted_wheel:{name}" for name in sorted(unlisted_wheels))
    unique_failures = tuple(dict.fromkeys(failures))
    return VerificationReport(
        passed=not unique_failures,
        failures=unique_failures,
        verified_files=tuple(sorted(verified)),
    )


def require_safe_runtime_weight(path: Path, approved_root: Path) -> Path:
    """Allow only recursively contained regular SafeTensors in notebook runtime."""
    root = approved_root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError("weight must be a contained regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("weight must be contained by the approved root") from exc
    if not resolved.is_file():
        raise ValueError("weight must be a contained regular file")
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError("runtime accepts .safetensors only")
    return resolved


_EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+")
_SHA256_OPTION = re.compile(r"(?:^|\s)--hash=sha256:[0-9a-f]{64}(?:\s|$)")


def validate_hash_locked_requirements(text: str) -> tuple[str, ...]:
    """Return deterministic failures for an offline binary requirements lock."""
    failures: list[str] = []
    logical_line = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        logical_line += 1
        prefix = f"line_{logical_line}"
        lowered = line.lower()
        if (
            "://" in lowered
            or lowered.startswith(("git+", "hg+", "svn+", "bzr+"))
            or " @ " in lowered
        ):
            failures.append(f"{prefix}:vcs_or_url")
        if _EXACT_REQUIREMENT.match(line) is None:
            failures.append(f"{prefix}:not_exactly_pinned")
        if _SHA256_OPTION.search(line) is None:
            failures.append(f"{prefix}:missing_sha256")
    return tuple(failures)
