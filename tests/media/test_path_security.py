from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.path_security import resolve_safe_media


def test_resolve_safe_media_accepts_nested_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    media = root / "nested" / "video.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video")
    safe = resolve_safe_media(media, (root,))
    assert safe.path == media.resolve()
    assert safe.bytes == 5
    assert safe.device >= 0
    assert safe.inode >= 0


def test_resolve_safe_media_rejects_file_outside_roots(tmp_path: Path) -> None:
    """Catches an absolute path bypassing the attached-Dataset allowlist."""
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    with pytest.raises(ValueError, match="approved root"):
        resolve_safe_media(outside, (root,))


def test_resolve_safe_media_rejects_symlink_traversal(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "video.mp4").write_bytes(b"video")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not permit creating test symlinks")
    with pytest.raises(ValueError, match=r"link|reparse|approved root"):
        resolve_safe_media(link / "video.mp4", (root,))
