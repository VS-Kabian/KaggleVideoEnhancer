"""Freeze one visible selectable preview candidate."""

from __future__ import annotations

from engvit.recipes.routing import RecipeCandidate
from engvit.types import Recipe


def freeze_recipe(candidate: RecipeCandidate) -> Recipe:
    if not candidate.selectable:
        raise ValueError(f"candidate {candidate.candidate_id} is not selectable")
    if candidate.model_id is None:
        raise ValueError("selectable recipe requires a model/tool ID")
    return Recipe(
        recipe_id=candidate.candidate_id,
        model_id=candidate.model_id,
        model_scale=candidate.model_scale,
        denoise_strength=None,
        fps_policy="rife" if candidate.kind == "interpolation" else "source_cfr",
        final_resize=None,
    )

