from __future__ import annotations

import pytest

from engvit.recipes.freeze import freeze_recipe
from engvit.recipes.routing import RecipeCandidate


def test_only_selectable_candidate_can_be_frozen() -> None:
    reference = RecipeCandidate(
        candidate_id="reference",
        label="Source reference",
        kind="reference",
        model_id=None,
        model_scale=1,
        selectable=False,
        estimated_cost="none",
        reason="comparison only",
    )
    with pytest.raises(ValueError, match="not selectable"):
        freeze_recipe(reference)
