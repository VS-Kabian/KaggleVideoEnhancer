from __future__ import annotations

import numpy as np

from engvit.quality.fidelity import psnr_rgb_u8


def test_psnr_is_infinite_for_identity_and_finite_for_degradation() -> None:
    reference = np.full((16, 16, 3), 128, dtype=np.uint8)
    degraded = reference.copy()
    degraded[0, 0] = 0
    assert psnr_rgb_u8(reference, reference) == float("inf")
    assert 0 < psnr_rgb_u8(reference, degraded) < float("inf")

