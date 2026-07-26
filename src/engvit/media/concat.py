"""Validated concat-demuxer assembly of homogeneous encoded segments."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from engvit.media.segments import scan_framehash
from engvit.orchestration.atomic import AtomicArtifactWriter
from engvit.types import ChunkCompletion, EncoderConfig, Rational

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class VideoArtifact:
    path: Path
    bytes: int
    sha256: str
    frame_count: int
    first_pts: int
    last_pts: int
    boundary_frame_hashes: tuple[str, ...]
    dimensions: tuple[int, int] | None = None
    time_base: Rational | None = None
    sar: Rational | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_concat(
    completions: tuple[ChunkCompletion, ...],
    config: EncoderConfig,
    *,
    output_path: Path,
    ffmpeg_path: Path,
) -> VideoArtifact:
    """Hash-check homogeneous segments, concat-copy, then scan every frame."""
    if not completions:
        raise ValueError("concat requires at least one completion")
    ordered = tuple(sorted(completions, key=lambda item: item.first_pts))
    expected_start = 0
    parent = ordered[0].partial_path.resolve(strict=True).parent
    for completion in ordered:
        path = completion.partial_path.resolve(strict=True)
        if path.parent != parent or _SAFE_NAME.fullmatch(path.name) is None:
            raise ValueError("concat segments must share one safe generated directory")
        if path.stat().st_size != completion.bytes or _sha256(path) != completion.sha256:
            raise ValueError("concat segment hash or size does not match completion")
        if completion.encoder_extradata_sha256 != config.self_test_sha256:
            raise ValueError("concat segment encoder identity does not match")
        if completion.first_pts != expected_start:
            raise ValueError("concat completion PTS contain a gap or overlap")
        if completion.last_pts - completion.first_pts + 1 != completion.frame_count:
            raise ValueError("concat completion frame count and PTS disagree")
        expected_start = completion.last_pts + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_payload = (
        "ffconcat version 1.0\n"
        + "".join(f"file {item.partial_path.name}\n" for item in ordered)
    ).encode("utf-8")
    list_path = parent / "engvit-segments.ffconcat"
    AtomicArtifactWriter().write(list_path, list_payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".mkv.partial",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        completed = subprocess.run(
            [
                str(ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                str(list_path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-f",
                "matroska",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed with code {completed.returncode}: "
                f"{completed.stderr[-1000:]}"
            )
        scan = scan_framehash(temporary, ffmpeg_path)
        total_frames = sum(item.frame_count for item in ordered)
        if len(scan.frames) != total_frames:
            raise ValueError("assembled video decoded frame count does not match")
        if scan.time_base != config.output_time_base:
            raise ValueError("assembled video time base does not match")
        if tuple(item.pts for item in scan.frames) != tuple(range(total_frames)):
            raise ValueError("assembled video PTS contain a gap or overlap")
        os.replace(temporary, output_path)
        return VideoArtifact(
            path=output_path,
            bytes=output_path.stat().st_size,
            sha256=_sha256(output_path),
            frame_count=total_frames,
            first_pts=0,
            last_pts=total_frames - 1,
            boundary_frame_hashes=(
                scan.frames[0].sha256,
                scan.frames[-1].sha256,
            ),
            dimensions=scan.dimensions,
            time_base=scan.time_base,
            sar=scan.sar,
        )
    finally:
        temporary.unlink(missing_ok=True)
