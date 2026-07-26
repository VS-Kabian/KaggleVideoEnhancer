"""Late-bound Spandrel image-model adapter with an exact RGB uint8 contract."""

from __future__ import annotations

import importlib
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from engvit.models.registry import ModelArtifact


class DeviceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["cpu", "cuda"]
    index: int | None


class SpandrelEnhancer:
    def __init__(
        self,
        descriptor: Any,
        torch_module: Any,
        *,
        scale: int,
        device: str,
        precision: str,
    ) -> None:
        self._descriptor = descriptor
        self._torch = torch_module
        self._scale = scale
        self._device = device
        self._precision = precision

    @property
    def scale(self) -> int:
        return self._scale

    def enhance(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if (
            frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3
            or frame_rgb.shape[2] != 3
        ):
            raise ValueError("enhancer input must be HxWx3 RGB uint8")
        torch = self._torch
        tensor = (
            torch.from_numpy(np.ascontiguousarray(frame_rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.float32)
            / 255.0
        )
        if self._precision == "fp16":
            tensor = tensor.to(dtype=torch.float16)
        with torch.inference_mode():
            output = self._descriptor(tensor)
        expected = (
            1,
            3,
            frame_rgb.shape[0] * self._scale,
            frame_rgb.shape[1] * self._scale,
        )
        if tuple(output.shape) != expected:
            raise ValueError("model output shape violates the registry scale contract")
        result = (
            output.clamp(0, 1)
            .mul(255)
            .round()
            .to(dtype=torch.uint8)
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return cast(NDArray[np.uint8], result)

    def close(self) -> None:
        self._descriptor.cpu()
        if self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def load_enhancer(
    artifact: ModelArtifact,
    device: DeviceSpec,
    precision: str,
) -> SpandrelEnhancer:
    """Import the audited neural stack only after artifact resolution."""
    if precision not in artifact.precision:
        raise ValueError(f"{artifact.model_id} does not permit {precision}")
    try:
        torch = importlib.import_module("torch")
        spandrel = importlib.import_module("spandrel")
    except ImportError as exc:
        raise RuntimeError(
            "Torch/Spandrel is absent; attach and verify the audited offline wheelhouse"
        ) from exc
    device_name = (
        f"cuda:{device.index or 0}" if device.kind == "cuda" else "cpu"
    )
    loader = spandrel.ModelLoader(device=device_name)
    descriptor = loader.load_from_file(str(artifact.weight_path))
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise ValueError("model is not a Spandrel ImageModelDescriptor")
    if (
        descriptor.scale != artifact.scale
        or descriptor.input_channels != artifact.input_channels
        or descriptor.output_channels != artifact.output_channels
    ):
        raise ValueError("Spandrel descriptor violates the model registry contract")
    if precision == "fp16" and not descriptor.supports_half:
        raise ValueError("Spandrel descriptor does not support fp16")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    descriptor.to(device=device_name, dtype=dtype).eval()
    return SpandrelEnhancer(
        descriptor,
        torch,
        scale=artifact.scale,
        device=device_name,
        precision=precision,
    )

