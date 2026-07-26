"""Deterministic private continuation archives with per-file receipts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from engvit.orchestration.atomic import AtomicArtifactWriter
from engvit.paths import create_job_paths
from engvit.types import JobPaths


class ContinuationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    archive_path: Path
    archive_bytes: int = Field(ge=0)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuation_files(job: JobPaths) -> tuple[tuple[str, Path], ...]:
    records: list[tuple[str, Path]] = []
    for label, root in (("artifacts", job.artifacts), ("segments", job.segments)):
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("continuation must not contain symbolic links")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            records.append((f"{label}/{relative}", path))
    if not records:
        raise ValueError("continuation contains no verified artifacts or segments")
    return tuple(records)


def prepare_continuation(
    job: JobPaths,
    destination: Path,
) -> ContinuationReceipt:
    """Create a deterministic archive; caller must store it in a private target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = _continuation_files(job)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".zip.partial",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    files: dict[str, str] = {}
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name, path in records:
                payload = path.read_bytes()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload)
                files[name] = _sha256_bytes(payload)
        payload = temporary.read_bytes()
        AtomicArtifactWriter().write(destination, payload)
    finally:
        temporary.unlink(missing_ok=True)
    return ContinuationReceipt(
        archive_path=destination.resolve(strict=True),
        archive_bytes=destination.stat().st_size,
        archive_sha256=_sha256_path(destination),
        files=files,
    )


def _safe_member(name: str) -> tuple[str, PurePosixPath]:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or len(member.parts) < 2:
        raise ValueError("continuation archive contains an unsafe member")
    category = member.parts[0]
    if category not in {"artifacts", "segments"}:
        raise ValueError("continuation archive contains an unknown category")
    return category, PurePosixPath(*member.parts[1:])


def resume_continuation(
    receipt: ContinuationReceipt,
    *,
    attached_root: Path,
    output_root: Path,
    job_id: str,
) -> JobPaths:
    """Verify an attached private archive before recreating the job layout."""
    root = attached_root.resolve(strict=True)
    candidate = receipt.archive_path
    if not candidate.exists():
        candidate = root / receipt.archive_path.name
    archive_path = candidate.resolve(strict=True)
    try:
        archive_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("continuation archive is outside the attached root") from exc
    if (
        archive_path.stat().st_size != receipt.archive_bytes
        or _sha256_path(archive_path) != receipt.archive_sha256
    ):
        raise ValueError("continuation archive size or hash does not match receipt")

    destination_root = output_root.resolve(strict=False) / job_id
    if destination_root.exists() and any(destination_root.rglob("*")):
        raise ValueError("resume destination is not empty")
    paths = create_job_paths(output_root, job_id)
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        names = tuple(info.filename for info in archive.infolist() if not info.is_dir())
        if set(names) != set(receipt.files):
            raise ValueError("continuation members do not match receipt")
        for name in sorted(names):
            category, relative = _safe_member(name)
            payload = archive.read(name)
            if _sha256_bytes(payload) != receipt.files[name]:
                raise ValueError(f"continuation member hash mismatch: {name}")
            base = paths.artifacts if category == "artifacts" else paths.segments
            destination = base.joinpath(*relative.parts)
            resolved = destination.resolve(strict=False)
            try:
                resolved.relative_to(base.resolve(strict=True))
            except ValueError as exc:
                raise ValueError("continuation member escapes job root") from exc
            AtomicArtifactWriter().write(destination, payload)
    return paths

