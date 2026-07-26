from __future__ import annotations

from dataclasses import replace

import numpy as np

from engvit.analysis.proxy_scan import ProxyFrame, scan_proxy
from engvit.analysis.scenes import detect_scenes
from tests.analysis.helpers import timeline
from tests.analysis.test_proxy_scan import solid


def test_detect_scenes_preserves_both_sides_of_hard_cut() -> None:
    plan = timeline(20)
    frames = tuple(
        ProxyFrame(index, solid(0 if index < 10 else 255)) for index in range(20)
    )
    scan = scan_proxy(frames, plan, max_samples_per_second=10)
    scenes = detect_scenes(scan)
    assert tuple((scene.start_frame, scene.end_frame) for scene in scenes) == (
        (0, 10),
        (10, 20),
    )
    assert scenes[1].cut_score > 0.5


def test_repeat_signal_does_not_create_false_cut() -> None:
    plan = timeline(3)
    frames = tuple(ProxyFrame(index, solid(64)) for index in range(3))
    scan = scan_proxy(frames, plan, max_samples_per_second=10)
    altered = replace(
        scan,
        rows=tuple(
            replace(row, temporal_information=1.0, motion=1.0, repeat=True)
            for row in scan.rows
        ),
    )
    assert len(detect_scenes(altered)) == 1


def test_numpy_import_used_for_test_environment() -> None:
    assert np.__version__

