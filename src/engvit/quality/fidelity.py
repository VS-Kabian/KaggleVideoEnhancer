"""Explicit-domain synthetic-reference fidelity metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def psnr_rgb_u8(
    reference: NDArray[np.uint8],
    candidate: NDArray[np.uint8],
) -> float:
    """Compute RGB-domain PSNR with data range 255 and no hidden border crop."""
    if (
        reference.dtype != np.uint8
        or candidate.dtype != np.uint8
        or reference.shape != candidate.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
    ):
        raise ValueError("PSNR inputs must be same-shape HxWx3 RGB uint8")
    difference = (
        reference.astype(np.float64) - candidate.astype(np.float64)
    )
    mse = float(np.mean(difference * difference))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0**2) / mse)

