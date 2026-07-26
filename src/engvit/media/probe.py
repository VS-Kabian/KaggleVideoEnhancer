"""Safe ffprobe invocation and defensive conversion into domain types."""

from __future__ import annotations

import json
import re
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from engvit.media.selection import Selection
from engvit.types import MediaInfo, Rational, StreamInfo, VideoStreamInfo

_MATRIX_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _rational(value: object) -> Rational | None:
    if value is None:
        return None
    text = str(value).strip().replace(":", "/")
    if "/" not in text:
        return None
    numerator_text, denominator_text = text.split("/", 1)
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return None
    if denominator == 0 or numerator == 0:
        return None
    return Rational(numerator, denominator)


def _display_matrix(side_data: object) -> tuple[float, ...] | None:
    if not isinstance(side_data, list):
        return None
    for item in side_data:
        if not isinstance(item, dict) or item.get("side_data_type") != "Display Matrix":
            continue
        matrix = item.get("displaymatrix")
        if not isinstance(matrix, str):
            return None
        values: list[float] = []
        for line in matrix.splitlines():
            payload = line.split(":", 1)[-1]
            values.extend(float(value) for value in _MATRIX_NUMBER.findall(payload))
        return tuple(values) if len(values) == 9 else None
    return None


def _side_data_types(side_data: object) -> tuple[str, ...]:
    if not isinstance(side_data, list):
        return ()
    return tuple(
        str(item["side_data_type"])
        for item in side_data
        if isinstance(item, dict) and item.get("side_data_type") is not None
    )


def _metadata(stream: dict[str, Any]) -> dict[str, str]:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def _disposition(stream: dict[str, Any]) -> dict[str, bool]:
    values = stream.get("disposition")
    if not isinstance(values, dict):
        return {}
    return {str(key): bool(value) for key, value in values.items()}


def _base_stream(stream: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(stream)
    return {
        "index": int(stream["index"]),
        "codec_type": str(stream.get("codec_type", "unknown")),
        "codec_name": (
            str(stream["codec_name"]) if stream.get("codec_name") is not None else None
        ),
        "time_base": _rational(stream.get("time_base")),
        "start_pts": _optional_int(stream.get("start_pts")),
        "duration_pts": _optional_int(stream.get("duration_ts")),
        "disposition": _disposition(stream),
        "language": metadata.get("language"),
        "metadata": metadata,
    }


def parse_ffprobe(payload: dict[str, Any], selection: Selection) -> MediaInfo:
    """Parse ffprobe JSON without assuming optional fields are present."""
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise ValueError("ffprobe response does not contain a streams array")
    streams: list[StreamInfo] = []
    for raw in raw_streams:
        if not isinstance(raw, dict) or "index" not in raw:
            raise ValueError("ffprobe returned a malformed stream")
        base = _base_stream(raw)
        if raw.get("codec_type") == "video":
            side_data_types = _side_data_types(raw.get("side_data_list"))
            if any(
                "encryption" in item.casefold() for item in side_data_types
            ) or any(
                key.casefold() in {"enc_key_id", "encryption_scheme"}
                for key in base["metadata"]
            ):
                raise ValueError("encrypted video streams are unsupported")
            parsed_stream: StreamInfo = VideoStreamInfo(
                **base,
                coded_width=int(raw.get("width", 0)),
                coded_height=int(raw.get("height", 0)),
                sample_aspect_ratio=_rational(raw.get("sample_aspect_ratio")),
                avg_frame_rate=_rational(raw.get("avg_frame_rate")),
                real_frame_rate=_rational(raw.get("r_frame_rate")),
                pixel_format=(
                    str(raw["pix_fmt"]) if raw.get("pix_fmt") is not None else None
                ),
                bits_per_raw_sample=_optional_int(raw.get("bits_per_raw_sample")),
                field_order=(
                    str(raw["field_order"])
                    if raw.get("field_order") is not None
                    else None
                ),
                color_range=(
                    str(raw["color_range"]) if raw.get("color_range") is not None else None
                ),
                color_space=(
                    str(raw["color_space"]) if raw.get("color_space") is not None else None
                ),
                color_transfer=(
                    str(raw["color_transfer"])
                    if raw.get("color_transfer") is not None
                    else None
                ),
                color_primaries=(
                    str(raw["color_primaries"])
                    if raw.get("color_primaries") is not None
                    else None
                ),
                display_matrix=_display_matrix(raw.get("side_data_list")),
                side_data_types=side_data_types,
            )
        else:
            parsed_stream = StreamInfo(**base)
        streams.append(parsed_stream)

    selected = next(
        (
            stream
            for stream in streams
            if stream.index == selection.video_stream_index
            and isinstance(stream, VideoStreamInfo)
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected video stream is absent or is not video")

    raw_format = payload.get("format")
    format_data = raw_format if isinstance(raw_format, dict) else {}
    duration: Decimal | None = None
    if format_data.get("duration") not in (None, "N/A"):
        try:
            duration = Decimal(str(format_data["duration"]))
        except InvalidOperation:
            duration = None
    return MediaInfo(
        source=selection.canonical_path,
        source_sha256=selection.source_sha256,
        format_name=str(format_data.get("format_name", "unknown")),
        duration_seconds=duration,
        streams=tuple(streams),
        selected_video_index=selection.video_stream_index,
    )


def probe_media(
    selection: Selection,
    ffprobe_path: Path,
    *,
    timeout_seconds: int = 120,
) -> MediaInfo:
    """Run an explicitly selected ffprobe binary without a shell."""
    completed = subprocess.run(
        [
            str(ffprobe_path),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(selection.canonical_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        shell=False,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("ffprobe JSON root must be an object")
    return parse_ffprobe(payload, selection)
