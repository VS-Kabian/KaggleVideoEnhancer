from __future__ import annotations

from engvit.analysis.scenes import Scene
from engvit.models.temporal.windows import WindowPolicy, plan_windows


def _scene(scene_id: int, start: int, end: int) -> Scene:
    return Scene(
        scene_id=scene_id,
        start_frame=start,
        end_frame=end,
        cut_score=0.0,
        confidence="high",
    )


def test_windows_cover_each_core_frame_once_and_never_cross_cuts() -> None:
    scenes = (_scene(0, 0, 5), _scene(1, 5, 11))
    windows = plan_windows(
        scenes,
        frame_count=11,
        policy=WindowPolicy(
            core_frames=3,
            context_before=2,
            context_after=2,
            calibration_sha256="a" * 64,
        ),
    )

    cores = tuple(
        frame
        for window in windows
        for frame in range(window.core_start, window.core_end)
    )
    assert cores == tuple(range(11))
    assert all(
        (window.input_end <= 5 if window.scene_id == 0 else window.input_start >= 5)
        for window in windows
    )
    assert all(
        window.relative_core.start is not None
        and window.relative_core.stop is not None
        and window.relative_core.start >= 0
        and window.relative_core.stop <= window.input_end - window.input_start
        for window in windows
    )
