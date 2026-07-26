"""Fail-closed job and release quality decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from engvit.types import MetricEvidence


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_job_metrics: tuple[str, ...]
    required_release_protocols: tuple[str, ...]


class QualityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    reasons: tuple[str, ...]


def _decide(
    requirements: tuple[str, ...],
    evidence: tuple[MetricEvidence, ...],
    *,
    key: str,
) -> QualityDecision:
    failures: list[str] = []
    missing: list[str] = []
    for requirement in requirements:
        matching = tuple(
            item
            for item in evidence
            if getattr(item, key) == requirement
        )
        if any(item.state == "FAIL" for item in matching):
            failures.append(requirement)
        elif not matching or not any(item.state == "PASS" for item in matching):
            missing.append(requirement)
    if failures:
        return QualityDecision(
            state="FAIL",
            reasons=tuple(f"required_failed:{item}" for item in failures),
        )
    if missing:
        return QualityDecision(
            state="NOT_EVALUATED",
            reasons=tuple(f"required_missing_or_unset:{item}" for item in missing),
        )
    return QualityDecision(state="PASS", reasons=())


def job_decision(
    job_evidence: tuple[MetricEvidence, ...],
    qualified_recipe_evidence: tuple[MetricEvidence, ...],
    policy: QualityPolicy,
) -> QualityDecision:
    """Require job metrics plus pre-qualified release protocols."""
    job = _decide(policy.required_job_metrics, job_evidence, key="metric")
    if job.state != "PASS":
        return job
    return _decide(
        policy.required_release_protocols,
        qualified_recipe_evidence,
        key="protocol",
    )


def release_decision(
    evidence: tuple[MetricEvidence, ...],
    policy: QualityPolicy,
) -> QualityDecision:
    """Require every release protocol; diagnostic blind metrics cannot substitute."""
    return _decide(
        policy.required_release_protocols,
        evidence,
        key="protocol",
    )

