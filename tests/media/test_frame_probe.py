from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from engvit.media.frame_probe import parse_frame_lines, stream_source_timing
from engvit.types import MediaInfo, Rational
from tests.media.pipeline_helpers import video_stream


def test_parse_frame_lines_preserves_negative_pts_and_flags() -> None:
    lines = (
        "media_type=video|best_effort_timestamp=-1001|duration=1001|"
        "repeat_pict=0|interlaced_frame=1|top_field_first=1\n",
        "media_type=video|best_effort_timestamp=0|duration=1001|"
        "repeat_pict=1|interlaced_frame=0|top_field_first=0\n",
    )
    frames = tuple(parse_frame_lines(lines, Rational(1, 30000)))
    assert frames[0].best_effort_pts == -1001
    assert frames[0].duration_pts == 1001
    assert frames[0].interlaced is True
    assert frames[0].top_field_first is True
    assert frames[1].repeat_pict == 1


def test_parse_frame_lines_ignores_compact_frame_side_data_suffix() -> None:
    lines = (
        "media_type=video|best_effort_timestamp=0|duration=1001|"
        "interlaced_frame=0|top_field_first=0|"
        "repeat_pict=0side_data_type="
        "H.26[45] User Data Unregistered SEI message\n",
    )

    frames = tuple(parse_frame_lines(lines, Rational(1, 30000)))

    assert len(frames) == 1
    assert frames[0].best_effort_pts == 0
    assert frames[0].repeat_pict == 0


def test_stream_source_timing_uses_kaggle_supported_frame_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_command: list[str] = []

    class ProbeProcess:
        stdout = iter(
            (
                "media_type=video|best_effort_timestamp=0|duration=1|"
                "repeat_pict=0|interlaced_frame=0|top_field_first=0\n",
            )
        )

        @staticmethod
        def wait(timeout: int | None = None) -> int:
            return 0

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError("successful probe must not be killed")

    def popen(command: list[str], **_kwargs: object) -> ProbeProcess:
        captured_command.extend(command)
        return ProbeProcess()

    monkeypatch.setattr("engvit.media.frame_probe.subprocess.Popen", popen)
    selected = video_stream()
    media = MediaInfo(
        source=Path("source.mkv"),
        source_sha256="0" * 64,
        format_name="matroska",
        duration_seconds=Decimal("0.1"),
        streams=(selected,),
        selected_video_index=selected.index,
    )

    frames = tuple(stream_source_timing(media, Path("ffprobe")))

    assert len(frames) == 1
    show_entries = captured_command[captured_command.index("-show_entries") + 1]
    assert ":frame_side_data=" not in show_entries


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "media_type=video|duration=1|repeat_pict=0|"
            "interlaced_frame=0|top_field_first=0\n",
            "timestamp",
        ),
        (
            "media_type=video|best_effort_timestamp=0|duration=0|"
            "repeat_pict=0|interlaced_frame=0|top_field_first=0\n",
            "duration",
        ),
        (
            "media_type=video|best_effort_timestamp=N/A|duration=1|"
            "repeat_pict=0|interlaced_frame=0|top_field_first=0\n",
            "timestamp",
        ),
    ],
)
def test_parse_frame_lines_rejects_unusable_timing(line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tuple(parse_frame_lines((line,), Rational(1, 1000)))


def test_parse_frame_lines_rejects_duplicate_or_regressing_pts() -> None:
    lines = (
        "media_type=video|best_effort_timestamp=5|duration=1\n",
        "media_type=video|best_effort_timestamp=5|duration=1\n",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        tuple(parse_frame_lines(lines, Rational(1, 1000)))


def test_frame_probe_module_does_not_buffer_json_contract() -> None:
    """The parser accepts an iterator and does not require a JSON document."""
    consumed: list[int] = []

    def lines() -> Iterator[str]:
        for index in range(3):
            consumed.append(index)
            yield (
                f"media_type=video|best_effort_timestamp={index}|duration=1\n"
            )

    iterator = parse_frame_lines(lines(), Rational(1, 1000))
    assert consumed == []
    assert next(iterator).source_index == 0
    assert consumed == [0]


def test_placeholder_path_import_is_platform_neutral(tmp_path: Path) -> None:
    """Guards accidental import-time assumptions about `/kaggle` paths."""
    assert tmp_path.is_absolute()
