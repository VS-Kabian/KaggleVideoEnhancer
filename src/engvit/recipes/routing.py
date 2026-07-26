"""Deterministic candidate routing from evidence and release capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from engvit.recipes.catalog import ReleaseCapabilities
from engvit.types import DiagnosticReport


class RecipeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    label: str
    kind: Literal["reference", "lanczos", "neural", "interpolation"]
    model_id: str | None
    model_scale: int
    selectable: bool
    estimated_cost: str
    reason: str


def eligible_recipes(
    report: DiagnosticReport,
    capabilities: ReleaseCapabilities,
    *,
    approved_model_ids: tuple[str, ...],
    target_size: tuple[int, int],
) -> tuple[RecipeCandidate, ...]:
    """Expose only candidates supported by both artifacts and release evidence."""
    del report
    width, height = target_size
    eight_k_request = width > 3840 or height > 2160
    candidates = [
        RecipeCandidate(
            candidate_id="source-reference",
            label="Source reference",
            kind="reference",
            model_id=None,
            model_scale=1,
            selectable=False,
            estimated_cost="none",
            reason="comparison_only",
        ),
        RecipeCandidate(
            candidate_id="lanczos",
            label="Lanczos baseline",
            kind="lanczos",
            model_id="lanczos",
            model_scale=2,
            selectable=not eight_k_request or capabilities.eight_k,
            estimated_cost="low",
            reason=(
                "deterministic_phase0"
                if not eight_k_request
                else "requires_8k_release_capability"
            ),
        ),
    ]
    spatial_enabled = (
        capabilities.eight_k if eight_k_request else capabilities.four_k
    )
    if spatial_enabled:
        for model_id in sorted(approved_model_ids):
            scale = 2 if "x2" in model_id.casefold() else 4
            candidates.append(
                RecipeCandidate(
                    candidate_id=f"neural-{model_id}",
                    label=f"Neural spatial: {model_id}",
                    kind="neural",
                    model_id=model_id,
                    model_scale=scale,
                    selectable=True,
                    estimated_cost="measured_benchmark_required",
                    reason="approved_model_and_release_capability",
                )
            )
    if capabilities.rife:
        candidates.append(
            RecipeCandidate(
                candidate_id="rife",
                label="RIFE interpolation",
                kind="interpolation",
                model_id="rife",
                model_scale=1,
                selectable=True,
                estimated_cost="measured_benchmark_required",
                reason="rife_release_capability",
            )
        )
    return tuple(candidates)

