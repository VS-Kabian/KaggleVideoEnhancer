"""Fail-closed privacy checks before sensitive media access."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Visibility = Literal["private", "public", "unknown"]
DatasetRole = Literal["media", "weights", "wheels", "continuation", "output"]


class DatasetVisibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: str = Field(min_length=1)
    version: str = Field(min_length=1)
    role: DatasetRole
    visibility: Visibility


class KaggleContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    notebook_id: str = Field(min_length=1)
    notebook_visibility: Visibility
    internet_enabled: bool
    datasets: tuple[DatasetVisibility, ...]
    visibility_attestation: str | None = None


class PreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    failures: tuple[str, ...]
    attestation_used: bool
    facts: dict[str, str | bool | int]


class SensitiveMediaPreflight:
    """Evaluate only observable facts and an explicit unknown-state attestation."""

    def run(self, context: KaggleContext) -> PreflightResult:
        failures: list[str] = []
        unknown_visibility = False
        if context.internet_enabled:
            failures.append("internet_enabled")
        if context.notebook_visibility == "public":
            failures.append("notebook_public")
        elif context.notebook_visibility == "unknown":
            unknown_visibility = True

        for dataset in context.datasets:
            if dataset.visibility == "public":
                failures.append(f"dataset_public:{dataset.handle}")
            elif dataset.visibility == "unknown":
                unknown_visibility = True

        attestation = (context.visibility_attestation or "").strip()
        attestation_used = unknown_visibility and bool(attestation)
        if unknown_visibility and not attestation_used:
            if context.notebook_visibility == "unknown":
                failures.append("notebook_visibility_unknown")
            for dataset in context.datasets:
                if dataset.visibility == "unknown":
                    failures.append(f"dataset_visibility_unknown:{dataset.handle}")

        unique_failures = tuple(dict.fromkeys(failures))
        return PreflightResult(
            passed=not unique_failures,
            failures=unique_failures,
            attestation_used=attestation_used,
            facts={
                "notebook_id": context.notebook_id,
                "internet_enabled": context.internet_enabled,
                "dataset_count": len(context.datasets),
            },
        )
