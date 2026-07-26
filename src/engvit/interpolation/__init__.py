"""Optional frame-interpolation contracts; no model is enabled by default."""

from engvit.interpolation.timeline import (
    InterpolationPlan,
    InterpolationTick,
    plan_interpolation,
)

__all__ = ["InterpolationPlan", "InterpolationTick", "plan_interpolation"]
