"""Deterministic Network Interpolation over already-safe tensor sets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SafeTensorSet:
    tensors: dict[str, NDArray[np.generic]]


def _hash_tensors(tensors: dict[str, NDArray[np.generic]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode())
        digest.update(tensor.dtype.str.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MergedTensorSet(SafeTensorSet):
    sha256: str
    strength: Decimal


def merge_dni(
    general: SafeTensorSet,
    weak: SafeTensorSet,
    strength: Decimal,
) -> MergedTensorSet:
    """Interpolate identical float tensors on CPU FP32; require equal non-floats."""
    if not strength.is_finite() or strength < 0 or strength > 1:
        raise ValueError("DNI strength must be a finite value from zero to one")
    if general.tensors.keys() != weak.tensors.keys():
        raise ValueError("DNI tensor keys do not match")
    result: dict[str, NDArray[np.generic]] = {}
    factor = np.float32(strength)
    for name in sorted(general.tensors):
        left = np.asarray(general.tensors[name])
        right = np.asarray(weak.tensors[name])
        if left.shape != right.shape:
            raise ValueError(f"DNI tensor shape mismatch for {name}")
        if left.dtype != right.dtype:
            raise ValueError(f"DNI tensor dtype mismatch for {name}")
        if np.issubdtype(left.dtype, np.floating):
            left32 = left.astype(np.float32, copy=False)
            right32 = right.astype(np.float32, copy=False)
            result[name] = left32 * (np.float32(1) - factor) + right32 * factor
        else:
            if not np.array_equal(left, right):
                raise ValueError(f"DNI non-floating tensor mismatch for {name}")
            result[name] = left.copy()
    return MergedTensorSet(
        tensors=result,
        sha256=_hash_tensors(result),
        strength=strength,
    )

