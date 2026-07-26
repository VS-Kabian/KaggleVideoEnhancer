"""NanoVSR challenger admission boundary."""

from engvit.models.temporal.adapters import (
    TemporalModelLock,
    require_temporal_available,
)


def require_available(lock: TemporalModelLock) -> None:
    require_temporal_available(lock)
