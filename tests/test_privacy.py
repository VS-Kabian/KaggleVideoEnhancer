from __future__ import annotations

from engvit.privacy import DatasetVisibility, KaggleContext, SensitiveMediaPreflight


def private_context() -> KaggleContext:
    return KaggleContext(
        notebook_id="owner/engvit-private",
        notebook_visibility="private",
        internet_enabled=False,
        datasets=(
            DatasetVisibility(
                handle="owner/private-video",
                version="7",
                role="media",
                visibility="private",
            ),
            DatasetVisibility(
                handle="owner/private-weights",
                version="3",
                role="weights",
                visibility="private",
            ),
        ),
    )


def test_sensitive_media_preflight_passes_verified_private_context() -> None:
    result = SensitiveMediaPreflight().run(private_context())
    assert result.passed is True
    assert result.failures == ()
    assert result.attestation_used is False


def test_sensitive_media_preflight_fails_unknown_visibility_without_attestation() -> None:
    """Catches reading sensitive media when Kaggle visibility is not observable."""
    context = private_context().model_copy(update={"notebook_visibility": "unknown"})
    result = SensitiveMediaPreflight().run(context)
    assert result.passed is False
    assert "notebook_visibility_unknown" in result.failures


def test_sensitive_media_preflight_records_attestation_for_unknown_visibility() -> None:
    context = private_context().model_copy(
        update={
            "notebook_visibility": "unknown",
            "visibility_attestation": (
                "I verified this notebook and all attached resources are private."
            ),
        }
    )
    result = SensitiveMediaPreflight().run(context)
    assert result.passed is True
    assert result.attestation_used is True


def test_sensitive_media_preflight_never_allows_known_public_media() -> None:
    """Catches treating an attestation as an override for known public exposure."""
    context = private_context().model_copy(
        update={
            "datasets": (
                DatasetVisibility(
                    handle="owner/public-video",
                    version="1",
                    role="media",
                    visibility="public",
                ),
            ),
            "visibility_attestation": "I accept the risk.",
        }
    )
    result = SensitiveMediaPreflight().run(context)
    assert result.passed is False
    assert "dataset_public:owner/public-video" in result.failures


def test_sensitive_media_preflight_requires_internet_disabled() -> None:
    """Catches reading private media in a network-enabled notebook."""
    context = private_context().model_copy(update={"internet_enabled": True})
    result = SensitiveMediaPreflight().run(context)
    assert result.passed is False
    assert "internet_enabled" in result.failures
