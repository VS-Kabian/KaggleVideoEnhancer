from __future__ import annotations

import numpy as np


class NearestEnhancer:
    def __init__(self, scale: int = 2, max_side: int | None = None) -> None:
        self._scale = scale
        self._max_side = max_side

    @property
    def scale(self) -> int:
        return self._scale

    def enhance(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._max_side is not None and max(frame_rgb.shape[:2]) > self._max_side:
            raise RuntimeError("CUDA out of memory")
        return np.repeat(
            np.repeat(frame_rgb, self._scale, axis=0),
            self._scale,
            axis=1,
        )

    def close(self) -> None:
        return None

