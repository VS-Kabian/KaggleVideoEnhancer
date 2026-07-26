from __future__ import annotations

import numpy as np

from engvit.tiling.calibrate import calibrate_tiling
from tests.tiling.helpers import NearestEnhancer


def test_calibration_reduces_only_oom_tile_and_records_attempts() -> None:
    corpus = (
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.full((96, 96, 3), 255, dtype=np.uint8),
    )
    result = calibrate_tiling(
        NearestEnhancer(scale=2, max_side=48),
        corpus,
        candidate_sizes=(64, 40, 32),
        context_pad=4,
        blend_overlap=8,
        precision="fp32",
        device_id=0,
    )
    assert result.policy.tile_size == 40
    assert tuple(item.state for item in result.attempts) == ("oom", "passed")
    assert result.policy.calibration_sha256 == result.sha256


def test_non_oom_runtime_error_is_not_hidden() -> None:
    class Broken(NearestEnhancer):
        def enhance(self, frame_rgb: np.ndarray) -> np.ndarray:
            raise RuntimeError("kernel bug")

    try:
        calibrate_tiling(
            Broken(),
            (np.zeros((32, 32, 3), dtype=np.uint8),),
            candidate_sizes=(32,),
            context_pad=0,
            blend_overlap=4,
            precision="fp32",
            device_id=0,
        )
    except RuntimeError as exc:
        assert "kernel bug" in str(exc)
    else:
        raise AssertionError("non-OOM failures must propagate")
