"""Non-executing structural inspection of SafeTensors files."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class SafeTensorHeader:
    tensors: dict[str, TensorDescriptor]
    metadata: dict[str, str]
    data_bytes: int


def inspect_safetensors(path: Path) -> SafeTensorHeader:
    """Validate SafeTensors JSON, offsets, tensor byte sizes, and file extent."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("model weight must use the .safetensors format")
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors header length is truncated")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > _MAX_HEADER_BYTES:
            raise ValueError("safetensors header size is invalid")
        if 8 + header_bytes > file_bytes:
            raise ValueError("safetensors header exceeds file size")
        raw_header = handle.read(header_bytes)
    try:
        payload = json.loads(raw_header.decode("utf-8").rstrip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("safetensors header root must be an object")

    raw_metadata = payload.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise ValueError("safetensors metadata must contain strings only")
    metadata = {str(key): str(value) for key, value in raw_metadata.items()}
    tensors: dict[str, TensorDescriptor] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, raw in payload.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("safetensors tensor entry is malformed")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype for {name}")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"invalid safetensors shape or offset for {name}")
        start, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES[str(dtype)]
        if start < 0 or end < start or end - start != expected:
            raise ValueError(f"safetensors offset size does not match {name}")
        descriptor = TensorDescriptor(
            dtype=str(dtype),
            shape=tuple(shape),
            start=start,
            end=end,
        )
        tensors[name] = descriptor
        ranges.append((start, end, name))
    if not tensors:
        raise ValueError("safetensors file contains no tensors")
    ordered = sorted(ranges)
    cursor = 0
    for start, end, name in ordered:
        if start != cursor:
            raise ValueError(f"safetensors offsets overlap or contain a gap at {name}")
        cursor = end
    data_bytes = file_bytes - 8 - header_bytes
    if cursor != data_bytes:
        raise ValueError("safetensors data offset size does not match file size")
    return SafeTensorHeader(
        tensors=tensors,
        metadata=metadata,
        data_bytes=data_bytes,
    )

