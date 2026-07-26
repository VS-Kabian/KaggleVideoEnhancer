from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def absolute_roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        (tmp_path / "input").resolve(),
        (tmp_path / "weights").resolve(),
        (tmp_path / "wheels").resolve(),
        (tmp_path / "output").resolve(),
    )

