"""Real-ESRGAN structural-signature validation."""

from __future__ import annotations

from engvit.models.weights import SafeTensorHeader

TensorContract = tuple[str, str, tuple[int, ...]]


def validate_realesrgan_signature(
    header: SafeTensorHeader,
    required_tensors: tuple[TensorContract, ...],
) -> None:
    """Reject a checkpoint unless every architecture canary matches exactly."""
    if not required_tensors:
        raise ValueError("Real-ESRGAN signature requires tensor canaries")
    for name, dtype, shape in required_tensors:
        descriptor = header.tensors.get(name)
        if descriptor is None:
            raise ValueError(f"Real-ESRGAN signature is missing tensor {name}")
        if descriptor.dtype != dtype:
            raise ValueError(f"Real-ESRGAN tensor dtype mismatch for {name}")
        if descriptor.shape != shape:
            raise ValueError(f"Real-ESRGAN tensor shape mismatch for {name}")

