"""Versioned scene-cut decisions in normalized output coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from engvit.analysis.proxy_scan import ProxyScan

SCENE_THRESHOLD_VERSION = "engvit-scene-v1"
_CUT_THRESHOLD = 0.45


@dataclass(frozen=True)
class Scene:
    scene_id: int
    start_frame: int
    end_frame: int
    cut_score: float
    confidence: str
    threshold_version: str = SCENE_THRESHOLD_VERSION


def detect_scenes(scan: ProxyScan) -> tuple[Scene, ...]:
    """Detect conservative hard cuts and preserve both sides as scene bounds."""
    if not scan.rows:
        raise ValueError("scene detection requires proxy rows")
    boundaries: list[tuple[int, float, str]] = []
    previous = scan.rows[0]
    for row in scan.rows[1:]:
        luma_jump = abs(row.luma_mean - previous.luma_mean)
        cut_score = min(
            1.0,
            0.55 * row.motion
            + 0.25 * luma_jump
            + 0.20 * row.temporal_information,
        )
        if not row.repeat and cut_score >= _CUT_THRESHOLD:
            confidence = "high" if cut_score >= 0.75 else "medium"
            boundaries.append((row.output_index, cut_score, confidence))
        previous = row

    starts = [(0, 0.0, "not_applicable"), *boundaries]
    scenes: list[Scene] = []
    for scene_id, (start, score, confidence) in enumerate(starts):
        end = (
            starts[scene_id + 1][0]
            if scene_id + 1 < len(starts)
            else scan.output_frame_count
        )
        scenes.append(
            Scene(
                scene_id=scene_id,
                start_frame=start,
                end_frame=end,
                cut_score=score,
                confidence=confidence,
            )
        )
    return tuple(scenes)

