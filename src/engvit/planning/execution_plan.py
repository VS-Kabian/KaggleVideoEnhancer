"""Canonical frozen ExecutionPlan construction and binding checks."""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict

from engvit.canonical import canonical_sha256
from engvit.types import (
    BenchmarkResult,
    ChunkSpec,
    EncoderConfig,
    ExecutionPlan,
    GeometryPlan,
    Recipe,
    TilePolicy,
)


class BindingVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    failures: tuple[str, ...]


def freeze_execution_plan(
    *,
    source_sha256: str,
    selection_sha256: str,
    timeline_sha256: str,
    diagnostic_sha256: str,
    recipe: Recipe,
    geometry: GeometryPlan,
    tiles: tuple[TilePolicy, ...],
    encoder: EncoderConfig,
    chunks: tuple[ChunkSpec, ...],
    benchmark: BenchmarkResult,
    environment_sha256: str,
    required_disk_bytes: int,
    predicted_seconds: int,
    safety_seconds: int,
) -> ExecutionPlan:
    """Construct and self-hash a complete immutable execution request."""
    if not tiles or not chunks:
        raise ValueError("execution plan requires tile policies and chunks")
    provisional = ExecutionPlan(
        schema_version="1",
        source_sha256=source_sha256,
        selection_sha256=selection_sha256,
        timeline_sha256=timeline_sha256,
        diagnostic_sha256=diagnostic_sha256,
        recipe=recipe,
        geometry=geometry,
        tiles=tiles,
        encoder=encoder,
        chunks=chunks,
        benchmark=benchmark,
        environment_sha256=environment_sha256,
        required_disk_bytes=required_disk_bytes,
        predicted_seconds=predicted_seconds,
        safety_seconds=safety_seconds,
        identity_sha256="",
    )
    return replace(
        provisional,
        identity_sha256=canonical_sha256(provisional, projection="identity"),
    )


def verify_plan_bindings(
    plan: ExecutionPlan,
    *,
    source_sha256: str,
    environment_sha256: str,
) -> BindingVerification:
    """Recompute identity and compare the two live mutable boundaries."""
    failures: list[str] = []
    if plan.source_sha256 != source_sha256:
        failures.append("source_sha256_changed")
    if plan.environment_sha256 != environment_sha256:
        failures.append("environment_sha256_changed")
    expected = canonical_sha256(
        replace(plan, identity_sha256=""),
        projection="identity",
    )
    if expected != plan.identity_sha256:
        failures.append("execution_plan_identity_changed")
    return BindingVerification(passed=not failures, failures=tuple(failures))

