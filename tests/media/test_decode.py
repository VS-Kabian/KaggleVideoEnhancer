from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.decode import decode_normalized_frames
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from tests.media.pipeline_helpers import ffmpeg_path, make_video, pipeline_contract


@pytest.mark.integration
def test_real_decoder_yields_exact_core_count_shape_and_dtype(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_video(source)
    timeline, graph = pipeline_contract(source)
    chunk = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=1,
            context_after_frames=1,
            execution_identity_sha256="a" * 64,
        ),
    )[1]
    frames = tuple(
        decode_normalized_frames(
            source=source,
            video_stream_index=0,
            chunk=chunk,
            plan=timeline,
            filter_graph=graph,
            output_size=(64, 36),
            ffmpeg_path=ffmpeg_path(),
        )
    )
    assert len(frames) == 3
    assert all(frame.shape == (36, 64, 3) for frame in frames)
    assert all(frame.dtype.name == "uint8" for frame in frames)

