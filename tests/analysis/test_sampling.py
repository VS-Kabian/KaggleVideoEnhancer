from __future__ import annotations

from dataclasses import replace

from engvit.analysis.proxy_scan import ProxyFrame, scan_proxy
from engvit.analysis.sampling import select_samples
from engvit.analysis.scenes import detect_scenes
from tests.analysis.helpers import timeline
from tests.analysis.test_proxy_scan import solid


def test_sampling_is_hash_deterministic_and_covers_cut_sides_and_extremes() -> None:
    plan = timeline(30)
    frames = tuple(
        ProxyFrame(index, solid(0 if index < 15 else 255)) for index in range(30)
    )
    scan = scan_proxy(frames, plan, max_samples_per_second=10)
    rows = [replace(row, banding_score=0.0) for row in scan.rows]
    rows[5] = replace(rows[5], noise_score=1.0)
    rows[25] = replace(rows[25], banding_score=1.0)
    scan = replace(scan, rows=tuple(rows))
    scenes = detect_scenes(scan)
    first = select_samples(scan, scenes, "a" * 64, count=12)
    second = select_samples(scan, scenes, "a" * 64, count=12)
    assert first == second
    assert 14 in first
    assert 15 in first
    assert 5 in first
    assert 25 in first
    assert first == tuple(sorted(first))


def test_sampling_rejects_count_too_small_for_mandatory_strata() -> None:
    plan = timeline(10)
    scan = scan_proxy(
        tuple(ProxyFrame(index, solid(index)) for index in range(10)),
        plan,
        max_samples_per_second=10,
    )
    scenes = detect_scenes(scan)
    try:
        select_samples(scan, scenes, "a" * 64, count=1)
    except ValueError as exc:
        assert "mandatory" in str(exc)
    else:
        raise AssertionError("undersized sample request must fail")
