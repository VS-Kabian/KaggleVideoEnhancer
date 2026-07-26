from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.encode import _encoder_command, encode_segment
from engvit.media.segments import scan_framehash
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from tests.analysis.helpers import timeline
from tests.media.pipeline_helpers import encoder_config, ffmpeg_path, rgb_frames


def test_encoder_command_uses_kaggle_compatible_passthrough_sync() -> None:
    command = _encoder_command(
        encoder_config(),
        (64, 36),
        Path("segment.mkv"),
        Path("ffmpeg"),
    )

    assert "-fps_mode" not in command
    assert command[command.index("-vsync") + 1] == "0"


@pytest.mark.integration
def test_real_encoder_writes_closed_segment_with_exact_decoded_pts(
    tmp_path: Path,
) -> None:
    chunk = plan_chunks(
        timeline(3),
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256="a" * 64,
        ),
    )[0]
    completion = encode_segment(
        frames=rgb_frames(3),
        chunk=chunk,
        lease_id="lease",
        config=encoder_config(),
        frame_size=(64, 36),
        output_path=tmp_path / "segment-000.mkv",
        ffmpeg_path=ffmpeg_path(),
    )
    hashes = scan_framehash(completion.partial_path, ffmpeg_path())
    assert completion.frame_count == 3
    assert tuple(item.pts for item in hashes.frames) == (0, 1, 2)
    assert tuple(item.duration for item in hashes.frames) == (1, 1, 1)
    assert completion.boundary_frame_hashes == (
        hashes.frames[0].sha256,
        hashes.frames[-1].sha256,
    )


def test_encoder_rejects_wrong_frame_count_before_commit(tmp_path: Path) -> None:
    chunk = plan_chunks(
        timeline(3),
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256="a" * 64,
        ),
    )[0]
    with pytest.raises(ValueError, match="frame count"):
        encode_segment(
            frames=rgb_frames(2),
            chunk=chunk,
            lease_id="lease",
            config=encoder_config(),
            frame_size=(64, 36),
            output_path=tmp_path / "segment.mkv",
            ffmpeg_path=ffmpeg_path(),
        )
