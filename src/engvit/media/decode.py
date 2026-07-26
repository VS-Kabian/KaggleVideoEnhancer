"""Exact raw-RGB decoding for a normalized chunk core."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from engvit.media.filters import FilterGraph
from engvit.types import ChunkSpec, TimelinePlan


def _read_exact(stream: IO[bytes], count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _decoder_command(
    *,
    source: Path,
    video_stream_index: int,
    input_options: tuple[str, ...],
    filters: tuple[str, ...],
    ffmpeg_path: Path,
) -> list[str]:
    return [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        *input_options,
        "-i",
        str(source),
        "-map",
        f"0:{video_stream_index}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-",
    ]


def decode_normalized_frames(
    *,
    source: Path,
    video_stream_index: int,
    chunk: ChunkSpec,
    plan: TimelinePlan,
    filter_graph: FilterGraph,
    output_size: tuple[int, int],
    ffmpeg_path: Path,
) -> Iterator[NDArray[np.uint8]]:
    """Decode exactly one logical core; no byte or frame mismatch is tolerated."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("raw output dimensions must be positive")
    expected = chunk.output_core_end - chunk.output_core_start
    if expected <= 0 or chunk.output_core_end > len(plan.output_frames):
        raise ValueError("chunk core is outside the timeline")
    filters = (
        *filter_graph.filters,
        f"scale=w={width}:h={height}:flags=lanczos",
        (
            f"trim=start_frame={chunk.output_core_start}:"
            f"end_frame={chunk.output_core_end}"
        ),
        "setpts=N",
    )
    command = _decoder_command(
        source=source,
        video_stream_index=video_stream_index,
        input_options=filter_graph.input_options,
        filters=filters,
        ffmpeg_path=ffmpeg_path,
    )
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=errors,
            shell=False,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError("FFmpeg decoder stdout was not created")
        try:
            for _ in range(expected):
                payload = _read_exact(process.stdout, frame_bytes)
                if len(payload) != frame_bytes:
                    raise ValueError("decoder returned a truncated raw RGB frame")
                yield np.frombuffer(payload, dtype=np.uint8).reshape(
                    (height, width, 3)
                ).copy()
            if process.stdout.read(1):
                raise ValueError("decoder returned more frames than the chunk core")
            return_code = process.wait()
            if return_code != 0:
                errors.seek(0)
                raise RuntimeError(
                    f"FFmpeg decoder failed with code {return_code}: "
                    f"{errors.read()[-1000:]}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
