from __future__ import annotations

from pathlib import Path

from engvit.kaggle import KagglePhase0Request
from engvit.privacy import KaggleContext, SensitiveMediaPreflight


def test_example_request_matches_current_schema() -> None:
    request = KagglePhase0Request.model_validate_json(
        Path("configs/kaggle-request.example.json").read_bytes()
    )

    assert request.target_width == 3840
    assert request.target_height == 2160


def test_example_private_context_fails_until_visibility_is_verified() -> None:
    context = KaggleContext.model_validate_json(
        Path("configs/kaggle-private-context.example.json").read_bytes()
    )

    result = SensitiveMediaPreflight().run(context)

    assert result.passed is False
    assert "notebook_visibility_unknown" in result.failures
