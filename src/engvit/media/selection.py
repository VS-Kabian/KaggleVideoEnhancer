"""Deterministic discovery and immutable selection of an input video."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engvit.media.path_security import resolve_safe_media

DatasetRole = Literal["media", "weights", "wheels", "models", "other"]
_VIDEO_EXTENSIONS = frozenset(
    {".avi", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm"}
)


class DatasetRoot(BaseModel):
    """One versioned, attached Kaggle Dataset and its trusted role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: str = Field(min_length=3)
    version: str = Field(min_length=1)
    root: Path
    role: DatasetRole


class MediaCandidate(BaseModel):
    """A display-safe media candidate relative to a known Dataset root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_handle: str
    dataset_version: str
    relative_path: str
    bytes: int = Field(ge=0)


class Selection(BaseModel):
    """Canonical, content-addressed input selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_handle: str
    dataset_version: str
    relative_path: str
    canonical_path: Path
    bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_stream_index: int = Field(ge=0)


def _sha256_verified(path: Path, expected_bytes: int, device: int, inode: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != expected_bytes
            or before.st_dev != device
            or before.st_ino != inode
        ):
            raise ValueError("media identity changed before hashing")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    final = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size)
    if (
        (after.st_dev, after.st_ino, after.st_size) != identity_before
        or (final.st_dev, final.st_ino, final.st_size) != identity_before
    ):
        raise ValueError("media identity changed while hashing")
    return digest.hexdigest()


def discover_media(datasets: tuple[DatasetRoot, ...]) -> tuple[MediaCandidate, ...]:
    """Enumerate regular video files from media-role Datasets only."""
    candidates: list[MediaCandidate] = []
    for dataset in datasets:
        if dataset.role != "media":
            continue
        root = dataset.root.resolve(strict=True)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if path.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            try:
                safe = resolve_safe_media(path, (root,))
            except ValueError:
                continue
            candidates.append(
                MediaCandidate(
                    dataset_handle=dataset.handle,
                    dataset_version=dataset.version,
                    relative_path=safe.path.relative_to(root).as_posix(),
                    bytes=safe.bytes,
                )
            )
    return tuple(candidates)


def select_media(
    candidate: MediaCandidate,
    dataset: DatasetRoot,
    *,
    video_stream_index: int,
) -> Selection:
    """Bind a candidate to its Dataset version and hash its complete contents."""
    if dataset.role != "media":
        raise ValueError("selection requires a media-role Dataset")
    if (
        candidate.dataset_handle != dataset.handle
        or candidate.dataset_version != dataset.version
    ):
        raise ValueError("candidate does not belong to the supplied Dataset version")
    relative = Path(candidate.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("candidate relative path is unsafe")
    safe = resolve_safe_media(dataset.root / relative, (dataset.root,))
    if safe.bytes != candidate.bytes:
        raise ValueError("media size changed after discovery")
    return Selection(
        dataset_handle=dataset.handle,
        dataset_version=dataset.version,
        relative_path=safe.path.relative_to(dataset.root.resolve(strict=True)).as_posix(),
        canonical_path=safe.path,
        bytes=safe.bytes,
        source_sha256=_sha256_verified(
            safe.path,
            expected_bytes=safe.bytes,
            device=safe.device,
            inode=safe.inode,
        ),
        video_stream_index=video_stream_index,
    )


def persist_selection(selection: Selection, destination: Path) -> None:
    """Atomically persist the immutable selection record as canonical JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        selection.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
