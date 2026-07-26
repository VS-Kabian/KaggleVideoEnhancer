from __future__ import annotations

from pathlib import Path

import pytest

from engvit.models.temporal.adapters import (
    load_temporal_lock,
    require_temporal_available,
)


@pytest.mark.parametrize(
    "filename",
    ("realbasicvsr.lock", "nanovsr.lock", "realviformer.lock"),
)
def test_repository_temporal_models_are_fail_closed(filename: str) -> None:
    root = Path(__file__).resolve().parents[3]
    lock = load_temporal_lock(root / "dependencies" / "temporal" / filename)

    with pytest.raises(RuntimeError, match="unavailable"):
        require_temporal_available(lock)
