from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.concat import validate_and_concat
from engvit.media.encode import encode_segment
from engvit.media.segments import scan_framehash
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from tests.analysis.helpers import timeline
from tests.media.pipeline_helpers import encoder_config, ffmpeg_path, rgb_frames


@pytest.mark.integration
def test_real_concat_has_exact_contiguous_frame_count_and_pts(tmp_path: Path) -> None:
    specs = plan_chunks(
        timeline(6),
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256="a" * 64,
        ),
    )
    completions = tuple(
        encode_segment(
            frames=rgb_frames(3, offset=index * 3),
            chunk=chunk,
            lease_id=f"lease-{index}",
            config=encoder_config(),
            frame_size=(64, 36),
            output_path=tmp_path / f"segment-{index:03d}.mkv",
            ffmpeg_path=ffmpeg_path(),
        )
        for index, chunk in enumerate(specs)
    )
    artifact = validate_and_concat(
        completions,
        encoder_config(),
        output_path=tmp_path / "joined.mkv",
        ffmpeg_path=ffmpeg_path(),
    )
    scan = scan_framehash(artifact.path, ffmpeg_path())
    assert artifact.frame_count == 6
    assert tuple(item.pts for item in scan.frames) == tuple(range(6))

