"""Recorded OOM-driven tile calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from engvit.canonical import canonical_sha256
from engvit.types import FrameEnhancer, TilePolicy


@dataclass(frozen=True)
class CalibrationAttempt:
    tile_size: int
    state: str
    corpus_frames: int
    message: str | None


@dataclass(frozen=True)
class TileCalibration:
    policy: TilePolicy
    attempts: tuple[CalibrationAttempt, ...]
    sha256: str


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return "out of memory" in message or "cuda error: memory" in message


def _center_crop(
    frame: NDArray[np.uint8],
    side: int,
) -> NDArray[np.uint8]:
    height, width = frame.shape[:2]
    crop_height = min(height, side)
    crop_width = min(width, side)
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    return frame[y0 : y0 + crop_height, x0 : x0 + crop_width]


def calibrate_tiling(
    enhancer: FrameEnhancer,
    corpus: tuple[NDArray[np.uint8], ...],
    *,
    candidate_sizes: tuple[int, ...],
    context_pad: int,
    blend_overlap: int,
    precision: str,
    device_id: int,
) -> TileCalibration:
    """Select the first fully passing tile size and preserve every OOM attempt."""
    if not corpus or not candidate_sizes:
        raise ValueError("tiling calibration requires corpus and candidate sizes")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("tiling precision must be fp32 or fp16")
    attempts: list[CalibrationAttempt] = []
    selected: int | None = None
    for tile_size in candidate_sizes:
        if tile_size <= blend_overlap:
            raise ValueError("calibration tile must be larger than blend overlap")
        try:
            inference_side = tile_size + 2 * context_pad
            for frame in corpus:
                crop = _center_crop(frame, inference_side)
                output = enhancer.enhance(crop)
                expected = (
                    crop.shape[0] * enhancer.scale,
                    crop.shape[1] * enhancer.scale,
                    3,
                )
                if output.dtype != np.uint8 or output.shape != expected:
                    raise ValueError("calibration model output violates scale contract")
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            attempts.append(
                CalibrationAttempt(
                    tile_size=tile_size,
                    state="oom",
                    corpus_frames=len(corpus),
                    message=str(exc),
                )
            )
            continue
        attempts.append(
            CalibrationAttempt(
                tile_size=tile_size,
                state="passed",
                corpus_frames=len(corpus),
                message=None,
            )
        )
        selected = tile_size
        break
    if selected is None:
        raise RuntimeError("all calibrated tile sizes exhausted GPU memory")
    provisional = TilePolicy(
        tile_size=selected,
        context_pad=context_pad,
        blend_overlap=blend_overlap,
        precision=precision,  # type: ignore[arg-type]
        device_id=device_id,
        calibration_sha256="0" * 64,
    )
    payload = {
        "policy": provisional,
        "attempts": tuple(attempts),
    }
    digest = canonical_sha256(payload, projection="identity")
    policy = replace(provisional, calibration_sha256=digest)
    return TileCalibration(
        policy=policy,
        attempts=tuple(attempts),
        sha256=digest,
    )

