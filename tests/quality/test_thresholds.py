from __future__ import annotations

from pathlib import Path

from engvit.quality.thresholds import load_thresholds


def test_workspace_thresholds_begin_unset() -> None:
    registry = load_thresholds(Path("quality/thresholds.yaml"))
    assert registry
    assert all(item.value is None and item.state == "UNSET" for item in registry)

