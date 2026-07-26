from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> str:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    complete = struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(payload)
    path.write_bytes(complete)
    return hashlib.sha256(complete).hexdigest()

