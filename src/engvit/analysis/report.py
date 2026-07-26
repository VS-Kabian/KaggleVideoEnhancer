"""Evidence-preserving aggregation of proxy analysis."""

from __future__ import annotations

from statistics import fmean

from engvit.analysis.proxy_scan import ProxyScan
from engvit.analysis.scenes import SCENE_THRESHOLD_VERSION, Scene
from engvit.types import DiagnosticReport, JSONValue, TimelinePlan


def build_diagnostic_report(
    *,
    source_sha256: str,
    timeline: TimelinePlan,
    scan: ProxyScan,
    scenes: tuple[Scene, ...],
    samples: tuple[int, ...],
) -> DiagnosticReport:
    """Build transparent aggregates while retaining raw-row and threshold IDs."""
    if scan.timeline_sha256 != timeline.sha256:
        raise ValueError("proxy scan does not belong to the supplied timeline")
    if any(index < 0 or index >= len(timeline.output_frames) for index in samples):
        raise ValueError("sample index is outside the timeline")
    rows = scan.rows
    features: dict[str, JSONValue] = {
        "feature_version": scan.feature_version,
        "scene_threshold_version": SCENE_THRESHOLD_VERSION,
        "proxy_row_count": len(rows),
        "scene_count": len(scenes),
        "mean_luma": fmean(row.luma_mean for row in rows),
        "mean_motion": fmean(row.motion for row in rows),
        "max_noise_score": max(row.noise_score for row in rows),
        "max_blocking_score": max(row.blocking_score for row in rows),
        "max_ringing_score": max(row.ringing_score for row in rows),
        "max_banding_score": max(row.banding_score for row in rows),
        "black_row_count": sum(row.black for row in rows),
        "freeze_row_count": sum(row.freeze for row in rows),
        "face_state": "NOT_EVALUATED",
    }
    warnings = (
        "face_detection_not_evaluated",
        "content_classification_is_heuristic",
    )
    return DiagnosticReport(
        source_sha256=source_sha256,
        timeline_sha256=timeline.sha256,
        scan_rows_sha256=scan.rows_sha256,
        sample_indexes=samples,
        features=features,
        warnings=warnings,
    )
