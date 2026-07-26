"""Final explicit video/ancillary remux command construction."""

from __future__ import annotations

import subprocess
from pathlib import Path

from engvit.media.streams import AncillaryStreamPlan


def build_remux_command(
    *,
    enhanced_video: Path,
    source_media: Path,
    destination: Path,
    stream_plan: AncillaryStreamPlan,
    ffmpeg_path: Path,
) -> tuple[str, ...]:
    """Build a non-shell command with explicit stream maps and no truncation."""
    return (
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(enhanced_video),
        "-i",
        str(source_media),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        *stream_plan.output_options,
        str(destination),
    )


def remux_ancillary(
    *,
    enhanced_video: Path,
    source_media: Path,
    destination: Path,
    stream_plan: AncillaryStreamPlan,
    ffmpeg_path: Path,
) -> Path:
    """Execute the frozen explicit remux plan."""
    subprocess.run(
        build_remux_command(
            enhanced_video=enhanced_video,
            source_media=source_media,
            destination=destination,
            stream_plan=stream_plan,
            ffmpeg_path=ffmpeg_path,
        ),
        check=True,
        shell=False,
    )
    return destination

