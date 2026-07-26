"""Streaming fixed-GOP segment encoding with decoded-frame validation."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from engvit.media.segments import scan_framehash
from engvit.types import ChunkCompletion, ChunkSpec, EncoderConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoder_command(
    config: EncoderConfig,
    frame_size: tuple[int, int],
    output: Path,
    ffmpeg_path: Path,
) -> list[str]:
    if config.encoder != "libx264" or config.codec != "h264":
        raise ValueError("Phase 0 segment encoder supports pinned libx264 only")
    mode = config.rate_control.get("mode")
    crf = config.rate_control.get("crf")
    if mode != "crf" or not isinstance(crf, int):
        raise ValueError("libx264 baseline requires integer CRF rate control")
    width, height = frame_size
    rate = f"{config.output_fps.numerator}/{config.output_fps.denominator}"
    time_base = (
        f"{config.output_time_base.numerator}:"
        f"{config.output_time_base.denominator}"
    )
    x264 = (
        f"open-gop=0:repeat-headers=1:keyint={config.gop}:"
        f"min-keyint={config.gop}:scenecut=0:bframes={config.b_frames}"
    )
    return [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        rate,
        "-i",
        "-",
        "-an",
        "-vf",
        "setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        config.preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        config.pixel_format,
        "-g",
        str(config.gop),
        "-keyint_min",
        str(config.gop),
        "-sc_threshold",
        "0",
        "-bf",
        str(config.b_frames),
        "-flags",
        "+cgop",
        "-x264-params",
        x264,
        "-fflags",
        "+bitexact",
        "-flags",
        "+bitexact",
        "-vsync",
        "0",
        "-enc_time_base",
        time_base,
        "-color_range",
        config.color["color_range"],
        "-colorspace",
        config.color["color_space"],
        "-color_trc",
        config.color["color_transfer"],
        "-color_primaries",
        config.color["color_primaries"],
        "-map_metadata",
        "-1",
        "-f",
        "matroska",
        str(output),
    ]


def encode_segment(
    *,
    frames: Iterable[NDArray[np.uint8]],
    chunk: ChunkSpec,
    lease_id: str,
    config: EncoderConfig,
    frame_size: tuple[int, int],
    output_path: Path,
    ffmpeg_path: Path,
) -> ChunkCompletion:
    """Stream one core to FFmpeg, atomically publish, then decode-validate it."""
    width, height = frame_size
    expected_count = chunk.output_core_end - chunk.output_core_start
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".mkv.partial",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    observed_count = 0
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
            process = subprocess.Popen(
                _encoder_command(config, frame_size, temporary, ffmpeg_path),
                stdin=subprocess.PIPE,
                stderr=errors,
                shell=False,
            )
            if process.stdin is None:
                process.kill()
                raise RuntimeError("FFmpeg encoder stdin was not created")
            try:
                for frame in frames:
                    if (
                        frame.dtype != np.uint8
                        or frame.shape != (height, width, 3)
                    ):
                        raise ValueError(
                            "encoder frames must match the declared RGB uint8 size"
                        )
                    if observed_count >= expected_count:
                        raise ValueError("encoder frame count exceeds chunk core")
                    process.stdin.write(frame.tobytes(order="C"))
                    observed_count += 1
                process.stdin.close()
                return_code = process.wait()
                if return_code != 0:
                    errors.seek(0)
                    raise RuntimeError(
                        f"FFmpeg encoder failed with code {return_code}: "
                        f"{errors.read()[-1000:]}"
                    )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if observed_count != expected_count:
            raise ValueError(
                f"encoder frame count {observed_count} does not match "
                f"chunk core {expected_count}"
            )
        scan = scan_framehash(temporary, ffmpeg_path)
        if len(scan.frames) != expected_count:
            raise ValueError("encoded segment decoded frame count does not match")
        if scan.time_base != config.output_time_base:
            raise ValueError("encoded segment time base does not match")
        if tuple(frame.pts for frame in scan.frames) != tuple(range(expected_count)):
            raise ValueError("encoded segment PTS are not contiguous from zero")
        if any(frame.duration != 1 for frame in scan.frames):
            raise ValueError("encoded segment frame durations are not one tick")
        os.replace(temporary, output_path)
        return ChunkCompletion(
            chunk_id=chunk.chunk_id,
            lease_id=lease_id,
            identity_sha256=chunk.identity_sha256,
            partial_path=output_path,
            bytes=output_path.stat().st_size,
            sha256=_sha256(output_path),
            frame_count=expected_count,
            first_pts=chunk.output_core_start,
            last_pts=chunk.output_core_end - 1,
            boundary_frame_hashes=(
                scan.frames[0].sha256,
                scan.frames[-1].sha256,
            ),
            encoder_extradata_sha256=config.self_test_sha256,
            observations={
                "decoded_time_base": (
                    f"{scan.time_base.numerator}/{scan.time_base.denominator}"
                ),
                "encoder": config.encoder,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
