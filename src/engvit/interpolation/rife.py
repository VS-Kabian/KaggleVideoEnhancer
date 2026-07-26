"""Fail-closed RIFE availability gate; Torch is never imported before approval."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class RifeLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    enabled: bool
    upstream_url: str
    commit: str | None
    code_archive_sha256: str | None
    license_id: str | None
    weight_filename: str | None
    weight_sha256: str | None
    reason: str


class RifeAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    reason: str


def load_rife_lock(path: Path) -> RifeLock:
    return RifeLock.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def assess_rife(lock: RifeLock) -> RifeAvailability:
    """Require every supply-chain field before any model import is allowed."""
    complete = all(
        (
            lock.commit,
            lock.code_archive_sha256,
            lock.license_id,
            lock.weight_filename,
            lock.weight_sha256,
        )
    )
    if not lock.enabled:
        return RifeAvailability(available=False, reason=lock.reason)
    if not complete:
        return RifeAvailability(
            available=False,
            reason="RIFE lock is incomplete",
        )
    return RifeAvailability(
        available=True,
        reason="lock complete; active-environment smoke still required",
    )
