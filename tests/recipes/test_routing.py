from __future__ import annotations

from engvit.recipes.catalog import ReleaseCapabilities
from engvit.recipes.routing import eligible_recipes
from engvit.types import DiagnosticReport


def report() -> DiagnosticReport:
    return DiagnosticReport(
        source_sha256="a" * 64,
        timeline_sha256="b" * 64,
        scan_rows_sha256="c" * 64,
        sample_indexes=(0, 1),
        features={"max_noise_score": 0.1},
        warnings=(),
    )


def test_fail_closed_capabilities_expose_reference_and_lanczos_only() -> None:
    candidates = eligible_recipes(
        report(),
        ReleaseCapabilities(
            schema_version="1",
            generated_from_acceptance_report=None,
            four_k=False,
            eight_k=False,
            rife=False,
            temporal_vsr=False,
            reason="not accepted",
        ),
        approved_model_ids=(),
        target_size=(3840, 2160),
    )
    assert tuple(item.kind for item in candidates) == ("reference", "lanczos")
    assert all(item.selectable == (item.kind == "lanczos") for item in candidates)


def test_neural_candidate_requires_capability_and_approved_model() -> None:
    capabilities = ReleaseCapabilities(
        schema_version="1",
        generated_from_acceptance_report="a" * 64,
        four_k=True,
        eight_k=False,
        rife=False,
        temporal_vsr=False,
        reason="4K accepted",
    )
    candidates = eligible_recipes(
        report(),
        capabilities,
        approved_model_ids=("realesrgan-x2plus",),
        target_size=(3840, 2160),
    )
    assert any(item.model_id == "realesrgan-x2plus" for item in candidates)
    assert not any(item.model_id == "rife" for item in candidates)

