from __future__ import annotations

import numpy as np

from engvit.tiling.runner import enhance_tiled
from engvit.types import TilePolicy
from tests.tiling.helpers import NearestEnhancer


def policy() -> TilePolicy:
    return TilePolicy(
        tile_size=24,
        context_pad=4,
        blend_overlap=7,
        precision="fp32",
        device_id=0,
        calibration_sha256="a" * 64,
    )


def test_tiled_identity_like_model_matches_untiled_on_odd_image() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(47, 61, 3), dtype=np.uint8)
    enhancer = NearestEnhancer(scale=2)
    tiled = enhance_tiled(frame, enhancer, policy())
    reference = enhancer.enhance(frame)
    np.testing.assert_array_equal(tiled, reference)
    assert tiled.shape == (94, 122, 3)


def test_tiled_runner_rejects_model_scale_violation() -> None:
    class WrongShape(NearestEnhancer):
        def enhance(self, frame_rgb: np.ndarray) -> np.ndarray:
            return frame_rgb

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        enhance_tiled(frame, WrongShape(scale=2), policy())
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("wrong-scale model output must fail")

