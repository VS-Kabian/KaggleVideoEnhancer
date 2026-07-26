from __future__ import annotations

from pathlib import Path

from engvit.media.concat import VideoArtifact
from engvit.quality.structural import run_structural_qa
from engvit.types import Rational


def artifact(frame_count: int = 6) -> VideoArtifact:
    return VideoArtifact(
        path=Path("video.mkv"),
        bytes=100,
        sha256="a" * 64,
        frame_count=frame_count,
        first_pts=0,
        last_pts=frame_count - 1,
        boundary_frame_hashes=("b" * 64, "c" * 64),
        dimensions=(64, 36),
        time_base=Rational(1, 10),
        sar=Rational(1, 1),
    )


def test_structural_qa_passes_exact_contract() -> None:
    evidence = run_structural_qa(
        artifact(),
        expected_frame_count=6,
        expected_dimensions=(64, 36),
        expected_time_base=Rational(1, 10),
    )
    assert all(item.state == "PASS" for item in evidence)


def test_truncation_and_wrong_geometry_fail_correct_metrics() -> None:
    evidence = run_structural_qa(
        artifact(5),
        expected_frame_count=6,
        expected_dimensions=(128, 72),
        expected_time_base=Rational(1, 10),
    )
    states = {item.metric: item.state for item in evidence}
    assert states["frame_count"] == "FAIL"
    assert states["dimensions"] == "FAIL"

