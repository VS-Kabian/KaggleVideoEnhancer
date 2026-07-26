from __future__ import annotations

from engvit.planning.session import AdmissionRequest, LiveResources, admit
from tests.planning.helpers import benchmark, recipe


def test_unknown_session_refuses_neural_but_not_short_baseline() -> None:
    live = LiveResources(
        free_disk_bytes=20_000_000_000,
        free_vram_bytes=8_000_000_000,
        remaining_session_seconds=None,
    )
    neural = admit(
        AdmissionRequest(
            recipe=recipe("realesrgan-x2plus"),
            benchmark=benchmark(),
            required_disk_bytes=5_000_000_000,
            predicted_seconds=1000,
            safety_seconds=300,
        ),
        live,
    )
    baseline = admit(
        AdmissionRequest(
            recipe=recipe(),
            benchmark=benchmark(),
            required_disk_bytes=5_000_000_000,
            predicted_seconds=100,
            safety_seconds=30,
        ),
        live,
    )
    assert neural.admitted is False
    assert "unknown" in neural.reasons[0]
    assert baseline.admitted is True
    assert baseline.warnings


def test_insufficient_disk_time_or_vram_refuses() -> None:
    decision = admit(
        AdmissionRequest(
            recipe=recipe("realesrgan-x2plus"),
            benchmark=benchmark(),
            required_disk_bytes=5_000,
            predicted_seconds=900,
            safety_seconds=200,
        ),
        LiveResources(
            free_disk_bytes=4_000,
            free_vram_bytes=1_000,
            remaining_session_seconds=1_000,
        ),
    )
    assert decision.admitted is False
    assert len(decision.reasons) == 3

