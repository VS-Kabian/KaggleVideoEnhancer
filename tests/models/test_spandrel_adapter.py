from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from engvit.models.registry import ModelArtifact
from engvit.models.spandrel_adapter import DeviceSpec, load_enhancer


def test_adapter_fails_clearly_when_audited_runtime_is_absent() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("test environment includes Torch")
    artifact = ModelArtifact(
        model_id="test",
        architecture="RRDBNet",
        scale=2,
        input_channels=3,
        output_channels=3,
        weight_path=Path("model.safetensors"),
        weight_sha256="a" * 64,
        weight_bytes=1,
        code_ref="ref",
        code_license="MIT",
        precision=("fp32",),
    )
    with pytest.raises(RuntimeError, match="offline wheelhouse"):
        load_enhancer(artifact, DeviceSpec(kind="cpu", index=None), "fp32")

