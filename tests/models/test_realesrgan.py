from __future__ import annotations

from engvit.models.realesrgan import validate_realesrgan_signature
from engvit.models.weights import SafeTensorHeader, TensorDescriptor


def test_realesrgan_signature_requires_declared_tensor_contract() -> None:
    header = SafeTensorHeader(
        tensors={
            "conv.weight": TensorDescriptor(
                dtype="F32",
                shape=(1, 1, 1, 1),
                start=0,
                end=4,
            )
        },
        metadata={},
        data_bytes=4,
    )
    validate_realesrgan_signature(
        header,
        (("conv.weight", "F32", (1, 1, 1, 1)),),
    )

