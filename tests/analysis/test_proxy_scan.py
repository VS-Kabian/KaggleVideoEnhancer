from __future__ import annotations

import numpy as np

from engvit.analysis.proxy_scan import ProxyFrame, scan_proxy
from tests.analysis.helpers import timeline


def solid(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_scan_proxy_covers_start_end_and_deterministic_cadence() -> None:
    plan = timeline(20)
    frames = (ProxyFrame(index, solid(index)) for index in range(20))
    scan = scan_proxy(frames, plan, max_samples_per_second=2)
    assert scan.sample_every_frames == 5
    assert tuple(row.output_index for row in scan.rows) == (0, 5, 10, 15, 19)
    assert scan.timeline_sha256 == plan.sha256
    assert scan.rows_sha256 != ""


def test_scan_proxy_detects_black_and_repeated_frames() -> None:
    plan = timeline(3)
    frames = tuple(ProxyFrame(index, solid(0)) for index in range(3))
    scan = scan_proxy(frames, plan, max_samples_per_second=10)
    assert scan.rows[0].black is True
    assert scan.rows[1].repeat is True
    assert scan.rows[1].freeze is True
    assert scan.rows[0].face_score is None
    assert scan.rows[0].face_state == "NOT_EVALUATED"


def test_scan_proxy_rejects_missing_or_noncanonical_coordinates() -> None:
    plan = timeline(3)
    frames = (ProxyFrame(0, solid(0)), ProxyFrame(2, solid(0)))
    try:
        scan_proxy(frames, plan, max_samples_per_second=10)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("missing proxy coordinates must fail")

