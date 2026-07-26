from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from engvit.media.segments import scan_framehash
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from engvit.orchestration.coordinator import Coordinator
from engvit.paths import create_job_paths
from engvit.pipeline import execute_neural_job
from engvit.types import Rational, TilePolicy
from tests.media.pipeline_helpers import (
    encoder_config,
    ffmpeg_path,
    make_video,
    pipeline_contract,
)


class NearestX2:
    @property
    def scale(self) -> int:
        return 2

    def enhance(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cast(
            NDArray[np.uint8],
            np.repeat(np.repeat(frame_rgb, 2, axis=0), 2, axis=1),
        )

    def close(self) -> None:
        return None


def test_approved_enhancer_path_uses_verified_chunk_and_concat_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    make_video(source)
    timeline, graph = pipeline_contract(source)
    identity = "e" * 64
    chunks = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256=identity,
        ),
    )
    paths = create_job_paths(tmp_path / "jobs", "neural-fixture")
    coordinator = Coordinator(
        manifest_path=paths.artifacts / "manifest.json",
        segments_root=paths.segments,
        chunks=chunks,
        job_identity_sha256=identity,
    )

    result = execute_neural_job(
        source=source,
        video_stream_index=0,
        timeline=timeline,
        filter_graph=graph,
        chunks=chunks,
        encoder=encoder_config(),
        input_size=(32, 18),
        output_size=(64, 36),
        enhancer=NearestX2(),
        tile_policy=TilePolicy(
            tile_size=16,
            context_pad=2,
            blend_overlap=4,
            precision="fp32",
            device_id=0,
            calibration_sha256="f" * 64,
        ),
        ffmpeg_path=ffmpeg_path(),
        coordinator=coordinator,
        output_path=paths.artifacts / "enhanced-video.mkv",
        worker_id="fixture-worker",
    )

    assert result.state == "complete"
    assert result.artifact is not None
    scan = scan_framehash(result.artifact.path, ffmpeg_path())
    assert len(scan.frames) == 6
    assert scan.dimensions == (64, 36)
    assert scan.sar == Rational(1, 1)
