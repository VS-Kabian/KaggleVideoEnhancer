from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engvit.media.selection import (
    DatasetRoot,
    discover_media,
    persist_selection,
    select_media,
)


def test_discover_media_filters_assets_and_preserves_dataset_version(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    weight_root = tmp_path / "weights"
    media_root.mkdir()
    weight_root.mkdir()
    (media_root / "clip.mp4").write_bytes(b"clip")
    (media_root / "notes.txt").write_text("not media", encoding="utf-8")
    (weight_root / "model.safetensors").write_bytes(b"weight")

    candidates = discover_media(
        (
            DatasetRoot(
                handle="owner/videos",
                version="4",
                root=media_root,
                role="media",
            ),
            DatasetRoot(
                handle="owner/weights",
                version="2",
                root=weight_root,
                role="weights",
            ),
        )
    )
    assert len(candidates) == 1
    assert candidates[0].dataset_handle == "owner/videos"
    assert candidates[0].dataset_version == "4"
    assert candidates[0].relative_path == "clip.mp4"


def test_select_media_computes_full_sha256(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    video = root / "clip.mkv"
    video.write_bytes(b"complete-video-content")
    dataset = DatasetRoot(
        handle="owner/videos",
        version="9",
        root=root,
        role="media",
    )
    candidate = discover_media((dataset,))[0]
    selection = select_media(candidate, dataset, video_stream_index=1)
    assert selection.source_sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert selection.video_stream_index == 1
    assert selection.canonical_path == video.resolve()


def test_select_media_rejects_file_changed_after_discovery(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    video = root / "clip.mkv"
    video.write_bytes(b"first")
    dataset = DatasetRoot(handle="owner/videos", version="9", root=root, role="media")
    candidate = discover_media((dataset,))[0]
    video.write_bytes(b"changed-length")
    with pytest.raises(ValueError, match="changed"):
        select_media(candidate, dataset, video_stream_index=0)


def test_persist_selection_writes_machine_readable_record(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    video = root / "clip.mkv"
    video.write_bytes(b"content")
    dataset = DatasetRoot(handle="owner/videos", version="9", root=root, role="media")
    selected = select_media(
        discover_media((dataset,))[0],
        dataset,
        video_stream_index=0,
    )
    destination = tmp_path / "job" / "selection.json"
    persist_selection(selected, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["source_sha256"] == selected.source_sha256
    assert payload["dataset_version"] == "9"
