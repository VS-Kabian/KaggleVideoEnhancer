"""Reference-free structural video evidence."""

from __future__ import annotations

from engvit.media.concat import VideoArtifact
from engvit.types import MetricEvidence, Rational


def _evidence(
    metric: str,
    passed: bool,
    value: object,
    reason: str | None,
) -> MetricEvidence:
    return MetricEvidence(
        evidence_id=f"structural:{metric}",
        protocol="structural",
        state="PASS" if passed else "FAIL",
        metric=metric,
        value=value,  # type: ignore[arg-type]
        threshold_version="structural-contract-v1",
        inputs={"artifact_sha256": "bound"},
        implementation={"engvit": "structural-v1"},
        reason=reason,
    )


def run_structural_qa(
    final: VideoArtifact,
    *,
    expected_frame_count: int,
    expected_dimensions: tuple[int, int],
    expected_time_base: Rational,
) -> tuple[MetricEvidence, ...]:
    """Check count, logical PTS, dimensions, time base, SAR, and boundaries."""
    frame_count_passed = final.frame_count == expected_frame_count
    pts_passed = (
        final.first_pts == 0
        and final.last_pts == expected_frame_count - 1
        and final.last_pts - final.first_pts + 1 == final.frame_count
    )
    dimensions_passed = final.dimensions == expected_dimensions
    time_base_passed = final.time_base == expected_time_base
    sar_passed = final.sar == Rational(1, 1)
    boundary_passed = bool(final.boundary_frame_hashes) and all(
        len(value) == 64 for value in final.boundary_frame_hashes
    )
    return (
        _evidence(
            "frame_count",
            frame_count_passed,
            final.frame_count,
            None if frame_count_passed else "decoded frame count mismatch",
        ),
        _evidence(
            "pts",
            pts_passed,
            {"first": final.first_pts, "last": final.last_pts},
            None if pts_passed else "logical PTS range is not contiguous",
        ),
        _evidence(
            "dimensions",
            dimensions_passed,
            list(final.dimensions) if final.dimensions is not None else None,
            None if dimensions_passed else "decoded dimensions mismatch",
        ),
        _evidence(
            "time_base",
            time_base_passed,
            (
                f"{final.time_base.numerator}/{final.time_base.denominator}"
                if final.time_base is not None
                else None
            ),
            None if time_base_passed else "decoded time base mismatch",
        ),
        _evidence(
            "sample_aspect_ratio",
            sar_passed,
            (
                f"{final.sar.numerator}/{final.sar.denominator}"
                if final.sar is not None
                else None
            ),
            None if sar_passed else "output SAR is not 1:1",
        ),
        _evidence(
            "boundary_hashes",
            boundary_passed,
            list(final.boundary_frame_hashes),
            None if boundary_passed else "boundary hash evidence is incomplete",
        ),
    )

