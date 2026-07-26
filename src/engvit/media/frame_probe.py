"""Bounded-memory extraction of complete source-frame timing."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from engvit.types import MediaInfo, Rational, SourceFrameTiming, VideoStreamInfo


def _fields(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in line.strip().split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # Some ffprobe compact writers concatenate the first nested
        # FRAME_SIDE_DATA field directly onto the final FRAME scalar.
        # The nested section is not part of the requested timing value.
        value = value.partition("side_data_type=")[0]
        parsed[key] = value
    return parsed


def _required_int(values: dict[str, str], key: str, label: str) -> int:
    value = values.get(key)
    if value in (None, "", "N/A"):
        raise ValueError(f"frame {label} is missing or unusable")
    assert value is not None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"frame {label} is missing or unusable") from exc


def _flag(values: dict[str, str], key: str) -> bool | None:
    value = values.get(key)
    if value in (None, "", "N/A"):
        return None
    if value not in {"0", "1"}:
        raise ValueError(f"frame {key} must be 0 or 1")
    return value == "1"


def parse_frame_lines(
    lines: Iterable[str],
    time_base: Rational,
) -> Iterator[SourceFrameTiming]:
    """Parse ffprobe compact lines lazily, requiring complete usable timing."""
    previous_pts: int | None = None
    source_index = 0
    for line in lines:
        if not line.strip():
            continue
        values = _fields(line)
        if values.get("media_type", "video") != "video":
            continue
        pts = _required_int(values, "best_effort_timestamp", "timestamp")
        duration_key = "duration" if "duration" in values else "pkt_duration"
        duration = _required_int(values, duration_key, "duration")
        if duration <= 0:
            raise ValueError("frame duration must be positive")
        if previous_pts is not None and pts <= previous_pts:
            raise ValueError("frame timestamps must be strictly increasing")
        repeat_pict = _required_int(
            {**values, "repeat_pict": values.get("repeat_pict", "0")},
            "repeat_pict",
            "repeat_pict",
        )
        interlaced = _flag(values, "interlaced_frame")
        top_field_first = _flag(values, "top_field_first")
        yield SourceFrameTiming(
            source_index=source_index,
            best_effort_pts=pts,
            duration_pts=duration,
            source_time_base=time_base,
            repeat_pict=repeat_pict,
            interlaced=bool(interlaced),
            top_field_first=top_field_first,
        )
        source_index += 1
        previous_pts = pts


def _selected_video(media: MediaInfo) -> VideoStreamInfo:
    selected = next(
        (
            stream
            for stream in media.streams
            if stream.index == media.selected_video_index
            and isinstance(stream, VideoStreamInfo)
        ),
        None,
    )
    if selected is None or selected.time_base is None:
        raise ValueError("selected video has no usable time base")
    return selected


def stream_source_timing(
    media: MediaInfo,
    ffprobe_path: Path,
    *,
    wait_timeout_seconds: int = 120,
) -> Iterator[SourceFrameTiming]:
    """Yield all selected-stream frames without materializing a JSON document."""
    video = _selected_video(media)
    time_base = video.time_base
    if time_base is None:
        raise ValueError("selected video has no usable time base")
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        str(video.index),
        "-show_frames",
        "-show_entries",
        (
            "frame=media_type,best_effort_timestamp,duration,pkt_duration,"
            "repeat_pict,interlaced_frame,top_field_first"
        ),
        "-of",
        "compact=p=0:nk=0",
        str(media.source),
    ]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            shell=False,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError("ffprobe stdout pipe was not created")
        try:
            yield from parse_frame_lines(process.stdout, time_base)
            return_code = process.wait(timeout=wait_timeout_seconds)
            if return_code != 0:
                errors.seek(0)
                stderr = errors.read()
                raise RuntimeError(
                    f"ffprobe frame scan failed with code {return_code}: "
                    f"{stderr[-1000:]}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
