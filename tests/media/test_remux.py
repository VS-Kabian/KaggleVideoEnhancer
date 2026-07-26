from __future__ import annotations

from pathlib import Path

from engvit.media.remux import build_remux_command
from engvit.media.streams import plan_ancillary_streams
from tests.media.pipeline_helpers import ffmpeg_path, unused_stream


def test_remux_command_maps_explicit_inputs_and_never_uses_shortest() -> None:
    command = build_remux_command(
        enhanced_video=Path("enhanced.mkv"),
        source_media=Path("source.mkv"),
        destination=Path("final.mp4"),
        stream_plan=plan_ancillary_streams(
            (unused_stream(),),
            "mp4_compatibility",
        ),
        ffmpeg_path=ffmpeg_path(),
    )
    assert "-shortest" not in command
    assert command.count("-map") >= 2
    assert "0:v:0" in command
    assert "1:1" in command
