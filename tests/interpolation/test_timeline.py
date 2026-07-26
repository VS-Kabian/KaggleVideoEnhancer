from __future__ import annotations

from engvit.analysis.scenes import Scene
from engvit.interpolation.timeline import plan_interpolation
from engvit.types import OutputFrameSpec, Rational, TimelinePlan


def _scene(scene_id: int, start: int, end: int) -> Scene:
    return Scene(
        scene_id=scene_id,
        start_frame=start,
        end_frame=end,
        cut_score=0.9 if start else 0.0,
        confidence="high" if start else "not_applicable",
    )


def _timeline(frame_count: int) -> TimelinePlan:
    return TimelinePlan(
        source_time_base=Rational(1, 10),
        output_time_base=Rational(1, 10),
        output_fps=Rational(10, 1),
        timing_transform=("source_cfr:10/1",),
        source_frames=(),
        output_frames=tuple(
            OutputFrameSpec(
                output_index=index,
                output_pts=index,
                output_duration=1,
                source_indexes=(index,),
                interpolation_fraction=None,
            )
            for index in range(frame_count)
        ),
        sha256="a" * 64,
    )


def test_half_open_ticks_have_exact_count_and_no_duplicates() -> None:
    timeline = _timeline(frame_count=6)

    plan = plan_interpolation(
        timeline,
        (_scene(0, 0, 6),),
        Rational(20, 1),
    )

    assert len(plan.ticks) == 12
    assert tuple(item.output_index for item in plan.ticks) == tuple(range(12))
    assert plan.output_time_base == Rational(1, 20)
    assert plan.ticks[1].fraction == Rational(1, 2)


def test_interpolation_never_pairs_frames_across_a_cut() -> None:
    timeline = _timeline(frame_count=6)

    plan = plan_interpolation(
        timeline,
        (_scene(0, 0, 3), _scene(1, 3, 6)),
        Rational(20, 1),
    )

    assert all(
        item.right_input_index is None
        or not (
            item.left_input_index < 3 <= item.right_input_index
        )
        for item in plan.ticks
    )
    before_cut = plan.ticks[5]
    cut_tick = plan.ticks[6]
    assert before_cut.left_input_index == 2
    assert before_cut.right_input_index is None
    assert cut_tick.left_input_index == 3
    assert cut_tick.scene_id == 1
