from __future__ import annotations

from dataclasses import replace

from engvit.planning.execution_plan import freeze_execution_plan, verify_plan_bindings
from tests.planning.helpers import benchmark, chunks, encoder, geometry, recipe, tile


def test_execution_plan_identity_binds_every_runtime_input() -> None:
    plan = freeze_execution_plan(
        source_sha256="1" * 64,
        selection_sha256="2" * 64,
        timeline_sha256="3" * 64,
        diagnostic_sha256="4" * 64,
        recipe=recipe(),
        geometry=geometry(),
        tiles=(tile(),),
        encoder=encoder(),
        chunks=chunks(),
        benchmark=benchmark(),
        environment_sha256="5" * 64,
        required_disk_bytes=10_000,
        predicted_seconds=100,
        safety_seconds=30,
    )
    mutated = freeze_execution_plan(
        source_sha256="9" * 64,
        selection_sha256="2" * 64,
        timeline_sha256="3" * 64,
        diagnostic_sha256="4" * 64,
        recipe=recipe(),
        geometry=geometry(),
        tiles=(tile(),),
        encoder=encoder(),
        chunks=chunks(),
        benchmark=benchmark(),
        environment_sha256="5" * 64,
        required_disk_bytes=10_000,
        predicted_seconds=100,
        safety_seconds=30,
    )
    assert plan.identity_sha256 != mutated.identity_sha256
    assert verify_plan_bindings(
        plan,
        source_sha256="1" * 64,
        environment_sha256="5" * 64,
    ).passed
    assert not verify_plan_bindings(
        replace(plan, environment_sha256="8" * 64),
        source_sha256="1" * 64,
        environment_sha256="5" * 64,
    ).passed

