from __future__ import annotations

import json
from pathlib import Path

from engvit.acceptance import (
    AcceptanceCase,
    AcceptanceMatrix,
    load_acceptance_matrix,
    release_capability_payload,
    run_acceptance,
)


def _matrix() -> AcceptanceMatrix:
    return AcceptanceMatrix(
        release_version="0.1.0",
        minimum_gpu_policy="P100 or exact faster recorded class",
        cases=(
            AcceptanceCase(
                case_id="phase0",
                description="deterministic baseline",
                receipt_filename="phase0.json",
                required_metrics=("structural",),
            ),
            AcceptanceCase(
                case_id="neural-x2",
                description="real x2 4K job",
                receipt_filename="neural-x2.json",
                required_metrics=("structural", "human"),
            ),
            AcceptanceCase(
                case_id="eight-k",
                description="exact 8K experiment",
                receipt_filename="eight-k.json",
                required_metrics=("encoder-main10",),
            ),
            AcceptanceCase(
                case_id="rife",
                description="RIFE timing",
                receipt_filename="rife.json",
                required_metrics=("temporal",),
            ),
            AcceptanceCase(
                case_id="temporal",
                description="temporal VSR challenger",
                receipt_filename="temporal.json",
                required_metrics=("temporal",),
            ),
        ),
        capability_requirements={
            "4k": ("phase0", "neural-x2"),
            "8k": ("phase0", "neural-x2", "eight-k"),
            "rife": ("phase0", "rife"),
            "temporal_vsr": ("phase0", "temporal"),
        },
    )


def _pass(case_id: str, metrics: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema_version": "1",
        "case_id": case_id,
        "state": "PASS",
        "environment_sha256": "a" * 64,
        "job_identity_sha256": "b" * 64,
        "artifact_sha256": "c" * 64,
        "metrics": {metric: "PASS" for metric in metrics},
        "reason": None,
        "independently_audited": True,
    }


def test_missing_receipts_keep_every_capability_false(tmp_path: Path) -> None:
    report = run_acceptance(_matrix(), evidence_root=tmp_path)

    assert report.capabilities == {
        "4k": False,
        "8k": False,
        "rife": False,
        "temporal_vsr": False,
    }
    assert all(item.state == "NOT_EVALUATED" for item in report.cases)


def test_incomplete_pass_receipt_is_a_failure(tmp_path: Path) -> None:
    (tmp_path / "phase0.json").write_text(
        json.dumps(_pass("phase0", ())),
        encoding="utf-8",
    )

    report = run_acceptance(_matrix(), evidence_root=tmp_path)

    phase0 = next(item for item in report.cases if item.case_id == "phase0")
    assert phase0.state == "FAIL"
    assert phase0.reason == "required_metrics_not_passed:structural"
    assert report.capabilities["4k"] is False


def test_only_capability_with_all_required_receipts_is_promoted(
    tmp_path: Path,
) -> None:
    (tmp_path / "phase0.json").write_text(
        json.dumps(_pass("phase0", ("structural",))),
        encoding="utf-8",
    )
    (tmp_path / "neural-x2.json").write_text(
        json.dumps(_pass("neural-x2", ("structural", "human"))),
        encoding="utf-8",
    )

    report = run_acceptance(_matrix(), evidence_root=tmp_path)
    payload = release_capability_payload(report)

    assert report.capabilities["4k"] is True
    assert report.capabilities["8k"] is False
    assert report.capabilities["rife"] is False
    assert payload["generated_from_acceptance_report"] == report.report_sha256


def test_repository_matrix_is_valid_and_fail_closed_without_receipts(
    tmp_path: Path,
) -> None:
    matrix = load_acceptance_matrix(Path("acceptance/matrix.yaml"))

    report = run_acceptance(matrix, evidence_root=tmp_path)

    assert matrix.release_version == "0.1.0"
    assert not any(report.capabilities.values())
