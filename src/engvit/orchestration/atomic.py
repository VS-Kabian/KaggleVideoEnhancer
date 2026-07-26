"""Durable same-directory atomic artifact replacement."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactReceipt:
    path: Path
    bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AtomicArtifactWriter:
    """Write, fsync, replace, and (on POSIX) fsync the containing directory."""

    def __init__(
        self,
        *,
        replace_attempts: int = 4,
        stage_hook: Callable[[str], None] | None = None,
    ) -> None:
        if replace_attempts < 1:
            raise ValueError("replace_attempts must be positive")
        self._replace_attempts = replace_attempts
        self._stage_hook = stage_hook

    def _stage(self, name: str) -> None:
        if self._stage_hook is not None:
            self._stage_hook(name)

    def write(
        self,
        path: Path,
        payload: bytes,
        *,
        expected_previous_sha256: str | None = None,
    ) -> ArtifactReceipt:
        path.parent.mkdir(parents=True, exist_ok=True)
        if expected_previous_sha256 is not None and (
            not path.is_file() or _sha256(path) != expected_previous_sha256
        ):
            raise ValueError("destination changed since the previous generation")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256(payload).hexdigest()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._stage("after_file_fsync")
            for attempt in range(self._replace_attempts):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt + 1 == self._replace_attempts:
                        raise
                    time.sleep(0.025 * (2**attempt))
            self._stage("after_replace")
            if os.name != "nt":
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            self._stage("after_directory_fsync")
            return ArtifactReceipt(path=path, bytes=len(payload), sha256=digest)
        finally:
            temporary.unlink(missing_ok=True)
