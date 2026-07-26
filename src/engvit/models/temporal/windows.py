"""Bounded temporal window/core/context planning with cut resets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from engvit.analysis.scenes import Scene


@dataclass(frozen=True)
class WindowPolicy:
    core_frames: int
    context_before: int
    context_after: int
    calibration_sha256: str


@dataclass(frozen=True)
class WindowSpec:
    scene_id: int
    input_start: int
    input_end: int
    core_start: int
    core_end: int

    @property
    def relative_core(self) -> slice:
        return slice(
            self.core_start - self.input_start,
            self.core_end - self.input_start,
        )


def plan_windows(
    scenes: tuple[Scene, ...],
    *,
    frame_count: int,
    policy: WindowPolicy,
) -> tuple[WindowSpec, ...]:
    """Cover every frame once while clamping all context to its scene."""
    if frame_count < 1:
        raise ValueError("temporal window planning requires frames")
    if (
        policy.core_frames < 1
        or policy.context_before < 0
        or policy.context_after < 0
        or len(policy.calibration_sha256) != 64
    ):
        raise ValueError("temporal window policy is invalid")
    ordered = tuple(sorted(scenes, key=lambda item: item.start_frame))
    if (
        not ordered
        or ordered[0].start_frame != 0
        or ordered[-1].end_frame != frame_count
        or any(
            left.end_frame != right.start_frame
            for left, right in pairwise(ordered)
        )
    ):
        raise ValueError("scenes must cover a contiguous full timeline")
    windows: list[WindowSpec] = []
    for scene in ordered:
        if scene.end_frame <= scene.start_frame:
            raise ValueError("scene bounds must be non-empty")
        core_start = scene.start_frame
        while core_start < scene.end_frame:
            core_end = min(core_start + policy.core_frames, scene.end_frame)
            windows.append(
                WindowSpec(
                    scene_id=scene.scene_id,
                    input_start=max(
                        scene.start_frame,
                        core_start - policy.context_before,
                    ),
                    input_end=min(
                        scene.end_frame,
                        core_end + policy.context_after,
                    ),
                    core_start=core_start,
                    core_end=core_end,
                )
            )
            core_start = core_end
    return tuple(windows)
