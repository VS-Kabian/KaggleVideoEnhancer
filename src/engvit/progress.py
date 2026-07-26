"""Bounded user-facing progress snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProgressSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str
    frames_complete: int = Field(ge=0)
    frames_total: int = Field(ge=1)
    chunks_complete: int = Field(ge=0)
    chunks_total: int = Field(ge=1)
    elapsed_seconds: int = Field(ge=0)
    eta_low_seconds: int | None = Field(default=None, ge=0)
    eta_high_seconds: int | None = Field(default=None, ge=0)
    confidence: Literal["low", "medium", "high", "unknown"]
    disk_used_bytes: int = Field(ge=0)
    disk_free_bytes: int = Field(ge=0)
    vram_used_bytes: int | None = Field(default=None, ge=0)
    vram_free_bytes: int | None = Field(default=None, ge=0)
    last_checkpoint: str | None
    retry_count: int = Field(ge=0)
    expected_session_finish: bool | None

    @model_validator(mode="after")
    def validate_bounds(self) -> ProgressSnapshot:
        if self.frames_complete > self.frames_total:
            raise ValueError("frames_complete exceeds frames_total")
        if self.chunks_complete > self.chunks_total:
            raise ValueError("chunks_complete exceeds chunks_total")
        if (
            self.eta_low_seconds is not None
            and self.eta_high_seconds is not None
            and self.eta_low_seconds > self.eta_high_seconds
        ):
            raise ValueError("ETA low bound exceeds high bound")
        return self

    @property
    def percent(self) -> float:
        return 100.0 * self.frames_complete / self.frames_total

