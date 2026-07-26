"""Separate code and model-weight license gates."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

WeightStatus = Literal["verified", "unverified", "prohibited"]


class LicenseRequirementError(ValueError):
    """The intended use is not permitted by recorded evidence."""


class IntendedUse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    private: bool
    commercial: bool
    redistribute: bool


class ModelLicenseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    code_license: str = Field(min_length=1)
    code_url: str = Field(min_length=1)
    weight_status: WeightStatus
    weight_terms_url: str | None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reviewed_by: str = Field(min_length=1)
    reviewed_on: date
    private_use: bool
    commercial_use: bool
    redistribution: bool
    notices: tuple[str, ...]

    @model_validator(mode="after")
    def verified_weight_has_hash_and_terms(self) -> ModelLicenseRecord:
        if self.weight_status == "verified":
            if self.weight_sha256 is None:
                raise ValueError("weight_sha256 is required when weight_status is verified")
            if not self.weight_terms_url:
                raise ValueError(
                    "weight_terms_url is required when weight_status is verified"
                )
        return self


class LicenseRegistry:
    def __init__(self, records: tuple[ModelLicenseRecord, ...]) -> None:
        by_id = {record.model_id: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("model license registry contains duplicate IDs")
        self._records = by_id

    @classmethod
    def from_yaml(cls, path: Path) -> LicenseRegistry:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        records = TypeAdapter(tuple[ModelLicenseRecord, ...]).validate_python(payload)
        return cls(records)

    def require_model(
        self,
        model_id: str,
        intended_use: IntendedUse,
    ) -> ModelLicenseRecord:
        try:
            record = self._records[model_id]
        except KeyError as exc:
            raise LicenseRequirementError(
                f"no license evidence for model {model_id}"
            ) from exc
        if record.weight_status != "verified":
            raise LicenseRequirementError(
                f"weight status for {model_id} is {record.weight_status}"
            )
        if intended_use.private and not record.private_use:
            raise LicenseRequirementError(f"{model_id} does not permit private use")
        if intended_use.commercial and not record.commercial_use:
            raise LicenseRequirementError(f"{model_id} does not permit commercial use")
        if intended_use.redistribute and not record.redistribution:
            raise LicenseRequirementError(f"{model_id} does not permit redistribution")
        return record
