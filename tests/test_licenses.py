from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from engvit.licenses import (
    IntendedUse,
    LicenseRegistry,
    LicenseRequirementError,
    ModelLicenseRecord,
)


def record(status: str = "verified", commercial: bool = True) -> ModelLicenseRecord:
    return ModelLicenseRecord(
        model_id="example-x2",
        code_license="BSD-3-Clause",
        code_url="https://example.invalid/code",
        weight_status=status,
        weight_terms_url="https://example.invalid/weights",
        weight_sha256="a" * 64,
        reviewed_by="reviewer",
        reviewed_on=date(2026, 7, 26),
        private_use=True,
        commercial_use=commercial,
        redistribution=False,
        notices=("Retain copyright notice.",),
    )


def test_license_registry_returns_verified_compatible_evidence() -> None:
    registry = LicenseRegistry((record(),))
    evidence = registry.require_model(
        "example-x2",
        IntendedUse(private=True, commercial=False, redistribute=False),
    )
    assert evidence.model_id == "example-x2"
    assert evidence.weight_status == "verified"


def test_license_registry_blocks_unverified_weights() -> None:
    """Catches confusing a repository code license with checkpoint permission."""
    registry = LicenseRegistry((record(status="unverified"),))
    with pytest.raises(LicenseRequirementError, match="unverified"):
        registry.require_model(
            "example-x2",
            IntendedUse(private=True, commercial=False, redistribute=False),
        )


def test_license_registry_blocks_incompatible_commercial_use() -> None:
    registry = LicenseRegistry((record(commercial=False),))
    with pytest.raises(LicenseRequirementError, match="commercial"):
        registry.require_model(
            "example-x2",
            IntendedUse(private=True, commercial=True, redistribute=False),
        )


def test_unverified_weight_may_record_unknown_hash_but_verified_may_not() -> None:
    unverified = record(status="unverified").model_copy(
        update={"weight_sha256": None}
    )
    assert unverified.weight_sha256 is None

    with pytest.raises(ValidationError, match="weight_sha256"):
        ModelLicenseRecord.model_validate(
            {
                **unverified.model_dump(),
                "weight_status": "verified",
                "weight_sha256": None,
            }
        )


def test_committed_realesrgan_weights_are_fail_closed_until_reviewed() -> None:
    """Catches publishing the baseline before checkpoint terms and hashes exist."""
    registry = LicenseRegistry.from_yaml(Path("licenses/model-weights.yaml"))
    with pytest.raises(LicenseRequirementError, match="unverified"):
        registry.require_model(
            "realesrgan-x2plus",
            IntendedUse(private=True, commercial=False, redistribute=False),
        )
