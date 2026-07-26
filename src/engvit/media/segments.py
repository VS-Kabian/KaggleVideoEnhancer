"""Decoded frame-hash evidence for segments and assembled video."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from engvit.types import Rational

_TIME_BASE = re.compile(r"^#tb\s+\d+:\s*(-?\d+)/(\d+)$")
_DIMENSIONS = re.compile(r"^#dimensions\s+\d+:\s*(\d+)x(\d+)$")
_SAR = re.compile(r"^#sar\s+\d+:\s*(-?\d+)/(\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrameHash:
    stream_index: int
    dts: int
    pts: int
    duration: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class FrameHashScan:
    time_base: Rational
    frames: tuple[FrameHash, ...]
    dimensions: tuple[int, int] | None = None
    sar: Rational | None = None


def parse_framehash(output: str) -> FrameHashScan:
    """Parse FFmpeg's framehash muxer output with exact integer timing."""
    time_base: Rational | None = None
    dimensions: tuple[int, int] | None = None
    sar: Rational | None = None
    frames: list[FrameHash] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = _TIME_BASE.match(line)
        if match is not None:
            time_base = Rational(int(match.group(1)), int(match.group(2)))
            continue
        dimension_match = _DIMENSIONS.match(line)
        if dimension_match is not None:
            dimensions = (
                int(dimension_match.group(1)),
                int(dimension_match.group(2)),
            )
            continue
        sar_match = _SAR.match(line)
        if sar_match is not None:
            sar = Rational(int(sar_match.group(1)), int(sar_match.group(2)))
            continue
        if not line or line.startswith("#"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 6 or _SHA256.fullmatch(fields[5]) is None:
            raise ValueError("malformed FFmpeg framehash row")
        frames.append(
            FrameHash(
                stream_index=int(fields[0]),
                dts=int(fields[1]),
                pts=int(fields[2]),
                duration=int(fields[3]),
                bytes=int(fields[4]),
                sha256=fields[5],
            )
        )
    if time_base is None:
        raise ValueError("FFmpeg framehash output has no time base")
    if not frames:
        raise ValueError("FFmpeg framehash output has no video frames")
    return FrameHashScan(
        time_base=time_base,
        frames=tuple(frames),
        dimensions=dimensions,
        sar=sar,
    )


def scan_framehash(path: Path, ffmpeg_path: Path) -> FrameHashScan:
    """Decode every video frame and collect PTS/duration/content hashes."""
    completed = subprocess.run(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return parse_framehash(completed.stdout)
