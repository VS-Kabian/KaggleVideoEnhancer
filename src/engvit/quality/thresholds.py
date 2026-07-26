"""Versioned threshold registry; unset values remain explicit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter


class MetricThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    state: Literal["UNSET", "CALIBRATED"]
    value: float | None
    unit: str
    version: str | None


def load_thresholds(path: Path) -> tuple[MetricThreshold, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TypeAdapter(tuple[MetricThreshold, ...]).validate_python(
        payload["thresholds"]
    )

