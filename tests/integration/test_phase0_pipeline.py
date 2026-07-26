from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.segments import scan_framehash
from engvit.orchestration.chunks import ChunkPolicy, plan_chunks
from engvit.orchestration.coordinator import Coordinator
from engvit.paths import create_job_paths
from engvit.pipeline import execute_lanczos_job
from engvit.types import Rational
from tests.media.pipeline_helpers import (
    encoder_config,
    ffmpeg_path,
    make_video,
    pipeline_contract,
)


@pytest.mark.integration
def test_phase0_job_runs_every_chunk_and_assembles_exact_video(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    make_video(source)
    timeline, graph = pipeline_contract(source)
    specs = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=1,
            context_after_frames=1,
            execution_identity_sha256="a" * 64,
        ),
    )
    paths = create_job_paths(tmp_path / "jobs", "phase0")
    service = Coordinator(
        manifest_path=paths.artifacts / "manifest.json",
        segments_root=paths.segments,
        chunks=specs,
        job_identity_sha256="a" * 64,
    )
    result = execute_lanczos_job(
        source=source,
        video_stream_index=0,
        timeline=timeline,
        filter_graph=graph,
        chunks=specs,
        encoder=encoder_config(),
        output_size=(64, 36),
        ffmpeg_path=ffmpeg_path(),
        coordinator=service,
        output_path=paths.artifacts / "enhanced-video.mkv",
        worker_id="integration-worker",
    )
    assert result.state == "complete"
    assert result.artifact is not None
    scan = scan_framehash(result.artifact.path, ffmpeg_path())
    assert tuple(frame.pts for frame in scan.frames) == tuple(range(6))
    assert scan.sar == Rational(1, 1)
    assert service.current.state == "complete"


@pytest.mark.integration
def test_phase0_job_pauses_after_commit_and_explicitly_resumes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    make_video(source)
    timeline, graph = pipeline_contract(source)
    specs = plan_chunks(
        timeline,
        (),
        ChunkPolicy(
            target_core_frames=3,
            max_scene_extension_frames=0,
            context_before_frames=0,
            context_after_frames=0,
            execution_identity_sha256="a" * 64,
        ),
    )
    paths = create_job_paths(tmp_path / "jobs", "resume")
    service = Coordinator(
        manifest_path=paths.artifacts / "manifest.json",
        segments_root=paths.segments,
        chunks=specs,
        job_identity_sha256="a" * 64,
    )
    arguments = {
        "source": source,
        "video_stream_index": 0,
        "timeline": timeline,
        "filter_graph": graph,
        "chunks": specs,
        "encoder": encoder_config(),
        "output_size": (64, 36),
        "ffmpeg_path": ffmpeg_path(),
        "coordinator": service,
        "output_path": paths.artifacts / "enhanced-video.mkv",
        "worker_id": "integration-worker",
    }
    paused = execute_lanczos_job(
        **arguments,  # type: ignore[arg-type]
        max_new_chunks=1,
    )
    assert paused.state == "paused"
    assert paused.artifact is None
    resumed = execute_lanczos_job(
        **arguments,  # type: ignore[arg-type]
        resume_paused=True,
    )
    assert resumed.state == "complete"
    assert resumed.artifact is not None
