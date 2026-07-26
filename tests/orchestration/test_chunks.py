from __future__ import annotations

from engvit.analysis.scenes import Scene
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from tests.analysis.helpers import timeline


def policy(identity: str = "a" * 64) -> ChunkPolicy:
    return ChunkPolicy(
        target_core_frames=4,
        max_scene_extension_frames=2,
        context_before_frames=1,
        context_after_frames=1,
        execution_identity_sha256=identity,
    )


def test_chunks_cover_timeline_once_and_prefer_near_scene_boundaries() -> None:
    plan = timeline(12)
    scenes = (
        Scene(0, 0, 5, 0.0, "not_applicable"),
        Scene(1, 5, 9, 0.8, "high"),
        Scene(2, 9, 12, 0.8, "high"),
    )
    chunks = plan_chunks(plan, scenes, policy())
    assert tuple((item.output_core_start, item.output_core_end) for item in chunks) == (
        (0, 5),
        (5, 9),
        (9, 12),
    )
    covered = [
        index
        for item in chunks
        for index in range(item.output_core_start, item.output_core_end)
    ]
    assert covered == list(range(12))
    assert chunks[1].context_before == 1
    assert chunks[1].context_after == 1
    assert chunks[1].source_decode_start_pts == 4
    assert chunks[1].source_decode_end_pts == 10


def test_execution_identity_invalidates_every_chunk_even_for_same_size_source() -> None:
    first = plan_chunks(timeline(12), (), policy("a" * 64))
    second = plan_chunks(timeline(12), (), policy("b" * 64))
    assert tuple(item.identity_sha256 for item in first) != tuple(
        item.identity_sha256 for item in second
    )

