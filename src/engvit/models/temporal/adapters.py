"""Shared fail-closed locks for isolated temporal research adapters."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class TemporalModelLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    model_id: str
    enabled: bool
    upstream_url: str
    commit: str | None
    code_archive_sha256: str | None
    license_id: str | None
    weight_sha256: str | None
    reason: str


def load_temporal_lock(path: Path) -> TemporalModelLock:
    return TemporalModelLock.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def require_temporal_available(lock: TemporalModelLock) -> None:
    """Stop before importing model code unless every approval exists."""
    if not lock.enabled:
        raise RuntimeError(f"{lock.model_id} unavailable: {lock.reason}")
    if not all(
        (
            lock.commit,
            lock.code_archive_sha256,
            lock.license_id,
            lock.weight_sha256,
        )
    ):
        raise RuntimeError(f"{lock.model_id} unavailable: incomplete lock")
