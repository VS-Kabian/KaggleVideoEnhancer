"""Runtime-neutral frame interpolation protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from engvit.types import Rational


class FrameInterpolator(Protocol):
    """A loaded interpolator operating on one same-scene frame pair."""

    def interpolate(
        self,
        left_rgb: NDArray[np.uint8],
        right_rgb: NDArray[np.uint8],
        fractions: tuple[Rational, ...],
    ) -> tuple[NDArray[np.uint8], ...]: ...

    def close(self) -> None: ...
