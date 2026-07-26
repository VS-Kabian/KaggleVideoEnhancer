"""Mechanical, fail-closed release acceptance aggregation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engvit.canonical import canonical_sha256

GateState = Literal["PASS", "FAIL", "NOT_EVALUATED"]
CapabilityName = Literal["4k", "8k", "rife", "temporal_vsr"]
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class AcceptanceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    description: str
    receipt_filename: str
    required_metrics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_safe_names(self) -> AcceptanceCase:
        if _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("acceptance case_id is unsafe")
        receipt = Path(self.receipt_filename)
        if (
            receipt.is_absolute()
            or ".." in receipt.parts
            or receipt.name != self.receipt_filename
            or receipt.suffix != ".json"
        ):
            raise ValueError("receipt_filename must be one safe JSON basename")
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("required_metrics must be unique")
        return self


class AcceptanceMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    release_version: str
    minimum_gpu_policy: str
    cases: tuple[AcceptanceCase, ...]
    capability_requirements: dict[CapabilityName, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_references(self) -> AcceptanceMatrix:
        case_ids = tuple(item.case_id for item in self.cases)
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("acceptance cases must be non-empty and unique")
        for capability in ("4k", "8k", "rife", "temporal_vsr"):
            required = self.capability_requirements.get(capability)
            if not required:
                raise ValueError(f"{capability} requires at least one acceptance case")
            unknown = set(required) - set(case_ids)
            if unknown:
                raise ValueError(
                    f"{capability} references unknown cases: {sorted(unknown)}"
                )
        return self


class AcceptanceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    case_id: str
    state: GateState
    environment_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    job_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    metrics: dict[str, GateState] = {}
    reason: str | None = None
    independently_audited: bool = False

    @model_validator(mode="after")
    def validate_pass_evidence(self) -> AcceptanceReceipt:
        if self.state == "PASS":
            if (
                self.environment_sha256 is None
                or self.job_identity_sha256 is None
                or self.artifact_sha256 is None
            ):
                raise ValueError("PASS receipt requires environment, job, and artifact hashes")
            if not self.independently_audited:
                raise ValueError("PASS receipt requires independent audit")
        elif not self.reason:
            raise ValueError("non-PASS receipt requires a reason")
        return self


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    state: GateState
    receipt_sha256: str | None
    reason: str


class AcceptanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    release_version: str
    matrix_sha256: str
    cases: tuple[CaseResult, ...]
    capabilities: dict[CapabilityName, bool]
    capability_reasons: dict[CapabilityName, tuple[str, ...]]
    report_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_acceptance_matrix(path: Path) -> AcceptanceMatrix:
    """Load a strict matrix; YAML is data only and cannot execute code."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AcceptanceMatrix.model_validate(payload)


def _evaluate_case(
    case: AcceptanceCase,
    evidence_root: Path,
) -> CaseResult:
    receipt_path = evidence_root / case.receipt_filename
    if not receipt_path.is_file():
        return CaseResult(
            case_id=case.case_id,
            state="NOT_EVALUATED",
            receipt_sha256=None,
            reason="receipt_missing",
        )
    receipt_hash = _sha256(receipt_path)
    try:
        receipt = AcceptanceReceipt.model_validate_json(receipt_path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError):
        return CaseResult(
            case_id=case.case_id,
            state="FAIL",
            receipt_sha256=receipt_hash,
            reason="receipt_invalid",
        )
    if receipt.case_id != case.case_id:
        return CaseResult(
            case_id=case.case_id,
            state="FAIL",
            receipt_sha256=receipt_hash,
            reason="receipt_case_id_mismatch",
        )
    missing = tuple(
        metric
        for metric in case.required_metrics
        if receipt.metrics.get(metric) != "PASS"
    )
    if missing:
        return CaseResult(
            case_id=case.case_id,
            state="FAIL" if receipt.state == "PASS" else receipt.state,
            receipt_sha256=receipt_hash,
            reason=f"required_metrics_not_passed:{','.join(missing)}",
        )
    return CaseResult(
        case_id=case.case_id,
        state=receipt.state,
        receipt_sha256=receipt_hash,
        reason=receipt.reason or "all_required_evidence_passed",
    )


def run_acceptance(
    matrix: AcceptanceMatrix,
    *,
    evidence_root: Path,
) -> AcceptanceReport:
    """Derive capabilities only from complete, strict, immutable receipts."""
    root = evidence_root.resolve(strict=False)
    results = tuple(_evaluate_case(case, root) for case in matrix.cases)
    by_id = {item.case_id: item for item in results}
    capabilities: dict[CapabilityName, bool] = {}
    reasons: dict[CapabilityName, tuple[str, ...]] = {}
    for capability, required_ids in matrix.capability_requirements.items():
        failed = tuple(
            f"{case_id}:{by_id[case_id].state}:{by_id[case_id].reason}"
            for case_id in required_ids
            if by_id[case_id].state != "PASS"
        )
        capabilities[capability] = not failed
        reasons[capability] = failed
    provisional = AcceptanceReport(
        release_version=matrix.release_version,
        matrix_sha256=canonical_sha256(matrix, projection="identity"),
        cases=results,
        capabilities=capabilities,
        capability_reasons=reasons,
        report_sha256="",
    )
    return provisional.model_copy(
        update={
            "report_sha256": canonical_sha256(
                provisional,
                projection="identity",
            )
        }
    )


def release_capability_payload(report: AcceptanceReport) -> dict[str, object]:
    """Return the exact public release-capabilities.json representation."""
    enabled = tuple(name for name, value in report.capabilities.items() if value)
    reason = (
        f"Enabled by acceptance report {report.report_sha256}: {', '.join(enabled)}"
        if enabled
        else "No capability has complete PASS evidence in the acceptance matrix."
    )
    return {
        "schema_version": "1",
        "generated_from_acceptance_report": report.report_sha256,
        "capabilities": dict(report.capabilities),
        "reason": reason,
    }
