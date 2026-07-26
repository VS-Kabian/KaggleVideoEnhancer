"""Fail-closed live resource admission."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engvit.types import BenchmarkResult, Recipe


class AdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    recipe: Recipe
    benchmark: BenchmarkResult
    required_disk_bytes: int = Field(ge=0)
    predicted_seconds: int = Field(ge=1)
    safety_seconds: int = Field(ge=0)


class LiveResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    free_disk_bytes: int = Field(ge=0)
    free_vram_bytes: int = Field(ge=0)
    remaining_session_seconds: int | None = Field(default=None, ge=0)


class AdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def admit(
    request: AdmissionRequest,
    live: LiveResources,
) -> AdmissionDecision:
    """Admit only when measured disk/VRAM/time satisfy the frozen request."""
    reasons: list[str] = []
    warnings: list[str] = []
    neural = request.recipe.model_id != "lanczos"
    if live.free_disk_bytes < request.required_disk_bytes:
        reasons.append("insufficient_free_disk")
    vram_required = (request.benchmark.peak_vram_bytes * 115 + 99) // 100
    if live.free_vram_bytes < vram_required:
        reasons.append("insufficient_free_vram_with_15_percent_reserve")
    required_time = request.predicted_seconds + request.safety_seconds
    if live.remaining_session_seconds is None:
        if neural:
            reasons.append("remaining_session_time_unknown_for_neural_job")
        else:
            warnings.append("remaining_session_time_unknown_baseline_may_pause")
    elif live.remaining_session_seconds < required_time:
        reasons.append("insufficient_remaining_session_time")
    return AdmissionDecision(
        admitted=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )

