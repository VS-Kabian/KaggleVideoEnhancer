from __future__ import annotations

from pathlib import Path

import pytest

from engvit.models.weights import inspect_safetensors
from tests.models.helpers import write_safetensors


def test_inspect_safetensors_validates_shapes_offsets_and_exact_data_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(
        path,
        {"conv.weight": ("F32", (1, 1, 1, 1), b"\0\0\0\0")},
    )
    header = inspect_safetensors(path)
    assert header.tensors["conv.weight"].shape == (1, 1, 1, 1)
    assert header.tensors["conv.weight"].dtype == "F32"
    assert header.data_bytes == 4


def test_inspect_safetensors_rejects_pickle_or_trailing_payload(
    tmp_path: Path,
) -> None:
    pickle_path = tmp_path / "model.pth"
    pickle_path.write_bytes(b"pickle")
    with pytest.raises(ValueError, match="safetensors"):
        inspect_safetensors(pickle_path)
    safe = tmp_path / "model.safetensors"
    write_safetensors(safe, {"x": ("F32", (1,), b"\0\0\0\0")})
    safe.write_bytes(safe.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match=r"size|offset"):
        inspect_safetensors(safe)
