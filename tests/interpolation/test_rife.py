from __future__ import annotations

from pathlib import Path

from engvit.interpolation.rife import assess_rife, load_rife_lock


def test_repository_rife_lock_is_explicitly_unavailable() -> None:
    root = Path(__file__).resolve().parents[2]
    lock = load_rife_lock(root / "dependencies" / "rife.lock")

    availability = assess_rife(lock)

    assert availability.available is False
    assert "Disabled" in availability.reason
