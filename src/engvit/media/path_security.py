"""Filesystem-boundary checks for untrusted media paths."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


@dataclass(frozen=True)
class SafeMediaPath:
    """A resolved regular file plus identity data captured from one stat call."""

    path: Path
    bytes: int
    device: int
    inode: int


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _lexical_relative(path: Path, root: Path) -> tuple[str, ...] | None:
    absolute_path = Path(os.path.abspath(path))
    absolute_root = Path(os.path.abspath(root))
    try:
        return absolute_path.relative_to(absolute_root).parts
    except ValueError:
        return None


def resolve_safe_media(path: Path, approved_roots: tuple[Path, ...]) -> SafeMediaPath:
    """Resolve a media path without permitting root escape or link traversal."""
    if not approved_roots:
        raise ValueError("at least one approved root is required")

    selected_root: Path | None = None
    relative_parts: tuple[str, ...] | None = None
    for root in approved_roots:
        parts = _lexical_relative(path, root)
        if parts is not None:
            selected_root = root
            relative_parts = parts
            break
    if selected_root is None or relative_parts is None:
        raise ValueError("media path is outside every approved root")

    current = Path(os.path.abspath(selected_root))
    for part in relative_parts:
        current /= part
        try:
            if _is_link_or_reparse(current):
                raise ValueError("media path must not traverse a link or reparse point")
        except OSError as exc:
            raise ValueError("media path does not resolve to a readable file") from exc

    try:
        resolved_root = selected_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError("media path is outside every approved root") from exc

    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("media path must identify a regular file")
    return SafeMediaPath(
        path=resolved,
        bytes=info.st_size,
        device=info.st_dev,
        inode=info.st_ino,
    )

