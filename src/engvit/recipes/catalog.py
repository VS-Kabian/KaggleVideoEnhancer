"""Mechanically loaded release capability record."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ReleaseCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    generated_from_acceptance_report: str | None
    four_k: bool
    eight_k: bool
    rife: bool
    temporal_vsr: bool
    reason: str

    @classmethod
    def from_json_file(cls, path: str | Path) -> ReleaseCapabilities:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        capabilities = payload.pop("capabilities")
        if not isinstance(capabilities, dict):
            raise ValueError("release capabilities field must be an object")
        return cls(
            schema_version=payload["schema_version"],
            generated_from_acceptance_report=payload[
                "generated_from_acceptance_report"
            ],
            four_k=bool(capabilities["4k"]),
            eight_k=bool(capabilities["8k"]),
            rife=bool(capabilities["rife"]),
            temporal_vsr=bool(capabilities["temporal_vsr"]),
            reason=payload["reason"],
        )

