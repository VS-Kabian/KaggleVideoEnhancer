from __future__ import annotations

from engvit.quality.decision import QualityPolicy, job_decision, release_decision
from engvit.types import MetricEvidence


def evidence(
    evidence_id: str,
    state: str,
    protocol: str = "structural",
) -> MetricEvidence:
    return MetricEvidence(
        evidence_id=evidence_id,
        protocol=protocol,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        metric=evidence_id,
        value=1,
        threshold_version="v1",
        inputs={},
        implementation={},
        reason=None,
    )


def test_missing_required_or_not_evaluated_never_passes() -> None:
    policy = QualityPolicy(
        required_job_metrics=("frame_count", "dimensions"),
        required_release_protocols=(
            "structural",
            "synthetic_hr_fidelity",
            "encoder_roundtrip",
        ),
    )
    decision = job_decision(
        (evidence("frame_count", "PASS"),),
        (),
        policy,
    )
    assert decision.state == "NOT_EVALUATED"


def test_any_required_failure_fails_release() -> None:
    policy = QualityPolicy(
        required_job_metrics=("frame_count",),
        required_release_protocols=("structural",),
    )
    decision = release_decision(
        (evidence("frame_count", "FAIL"),),
        policy,
    )
    assert decision.state == "FAIL"

