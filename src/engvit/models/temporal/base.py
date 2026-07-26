"""Temporal enhancement protocol kept separate from release recipes."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class TemporalEnhancer(Protocol):
    @property
    def scale(self) -> int: ...

    def enhance_window(
        self,
        frames_rgb: tuple[NDArray[np.uint8], ...],
        core: slice,
    ) -> tuple[NDArray[np.uint8], ...]: ...

    def close(self) -> None: ...
