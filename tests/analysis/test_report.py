from __future__ import annotations

from engvit.analysis.proxy_scan import ProxyFrame, scan_proxy
from engvit.analysis.report import build_diagnostic_report
from engvit.analysis.sampling import select_samples
from engvit.analysis.scenes import detect_scenes
from tests.analysis.helpers import timeline
from tests.analysis.test_proxy_scan import solid


def test_report_preserves_raw_hash_versions_and_low_confidence_warning() -> None:
    plan = timeline(12)
    scan = scan_proxy(
        tuple(ProxyFrame(index, solid(index * 10)) for index in range(12)),
        plan,
        max_samples_per_second=10,
    )
    scenes = detect_scenes(scan)
    samples = select_samples(scan, scenes, "a" * 64, count=8)
    report = build_diagnostic_report(
        source_sha256="a" * 64,
        timeline=plan,
        scan=scan,
        scenes=scenes,
        samples=samples,
    )
    assert report.scan_rows_sha256 == scan.rows_sha256
    assert report.sample_indexes == samples
    assert report.features["feature_version"] == "engvit-proxy-v1"
    assert "face_detection_not_evaluated" in report.warnings
