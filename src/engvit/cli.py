"""Privacy-conscious command line interface for the Kaggle workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import TextIO, TypeAlias

from pydantic import BaseModel

from engvit.kaggle import (
    KagglePhase0Request,
    PreparedPhase0,
    prepare_phase0,
    run_prepared_phase0,
)
from engvit.media.concat import VideoArtifact
from engvit.media.segments import scan_framehash
from engvit.media.selection import DatasetRoot, discover_media
from engvit.paths import create_job_paths
from engvit.persistence import prepare_continuation
from engvit.planning.benchmark import record_benchmark
from engvit.privacy import KaggleContext, SensitiveMediaPreflight
from engvit.quality.structural import run_structural_qa
from engvit.recipes.preview import PreviewItem, render_preview_html
from engvit.smoke import run_phase0_smoke

Handler: TypeAlias = Callable[[argparse.Namespace], tuple[int, object]]


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Decimal):
        return str(value)
    return value


def _emit(stream: TextIO, value: object) -> None:
    stream.write(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _read_model(path: Path, model: type[BaseModel]) -> BaseModel:
    return model.model_validate_json(path.read_bytes())


def _binary(requested: str | None, name: str) -> Path:
    candidate = requested or shutil.which(name)
    if candidate is None:
        raise FileNotFoundError(f"{name} is unavailable")
    return Path(candidate).resolve(strict=True)


def _request(path: Path) -> KagglePhase0Request:
    return KagglePhase0Request.model_validate_json(path.read_bytes())


def _prepared(args: argparse.Namespace) -> PreparedPhase0:
    return prepare_phase0(
        _request(args.request),
        ffmpeg_path=_binary(args.ffmpeg, "ffmpeg"),
        ffprobe_path=_binary(args.ffprobe, "ffprobe"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_from_existing(prepared: PreparedPhase0) -> VideoArtifact:
    path = prepared.paths.artifacts / "enhanced-video.mkv"
    if prepared.coordinator.current.state != "complete" or not path.is_file():
        raise ValueError("job is not complete; QA will not execute pending work")
    scan = scan_framehash(path, prepared.ffmpeg_path)
    if not scan.frames:
        raise ValueError("final artifact contains no decoded video frames")
    return VideoArtifact(
        path=path,
        bytes=path.stat().st_size,
        sha256=_sha256(path),
        frame_count=len(scan.frames),
        first_pts=scan.frames[0].pts,
        last_pts=scan.frames[-1].pts,
        boundary_frame_hashes=(scan.frames[0].sha256, scan.frames[-1].sha256),
        dimensions=scan.dimensions,
        time_base=scan.time_base,
        sar=scan.sar,
    )


def _add_request_binaries(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engvit",
        description="Fail-closed Kaggle video enhancement operator interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--context", type=Path, required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--dataset-root", type=Path, required=True)
    discover.add_argument("--handle", required=True)
    discover.add_argument("--version", required=True)

    for name in ("analyze", "preview", "pause", "resume", "qa"):
        _add_request_binaries(subparsers.add_parser(name))

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--output-root", type=Path, required=True)
    benchmark.add_argument("--ffmpeg")

    run = subparsers.add_parser("run")
    _add_request_binaries(run)
    run.add_argument("--max-new-chunks", type=int)
    run.add_argument("--resume-paused", action="store_true")

    persist = subparsers.add_parser("persist")
    persist.add_argument("--output-root", type=Path, required=True)
    persist.add_argument("--job-id", required=True)
    persist.add_argument("--archive", type=Path, required=True)
    return parser


def _preflight(args: argparse.Namespace) -> tuple[int, object]:
    context = _read_model(args.context, KaggleContext)
    assert isinstance(context, KaggleContext)
    result = SensitiveMediaPreflight().run(context)
    return (0 if result.passed else 2), result


def _discover(args: argparse.Namespace) -> tuple[int, object]:
    dataset = DatasetRoot(
        handle=args.handle,
        version=args.version,
        root=args.dataset_root,
        role="media",
    )
    candidates = discover_media((dataset,))
    return 0, {"count": len(candidates), "candidates": candidates}


def _analyze(args: argparse.Namespace) -> tuple[int, object]:
    prepared = _prepared(args)
    duration = prepared.media.duration_seconds
    return 0, {
        "state": "prepared",
        "source_sha256": prepared.selection.source_sha256,
        "duration_seconds": duration,
        "selected_video_index": prepared.selection.video_stream_index,
        "output_frames": len(prepared.timeline.output_frames),
        "output_fps": prepared.timeline.output_fps,
        "target_size": prepared.geometry.target_size,
        "color_state": prepared.color.state,
        "chunk_count": len(prepared.chunks),
        "execution_identity_sha256": prepared.execution_identity_sha256,
    }


def _preview(args: argparse.Namespace) -> tuple[int, object]:
    prepared = _prepared(args)
    video_name = "source-reference.mp4"
    crop_name = "source-reference-01.png"
    video_path = prepared.paths.previews / video_name
    crop_path = prepared.paths.previews / crop_name
    common = [
        str(prepared.ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(prepared.selection.canonical_path),
        "-map",
        f"0:{prepared.selection.video_stream_index}",
    ]
    subprocess.run(
        [
            *common,
            "-t",
            "5",
            "-vf",
            "scale=640:-2:flags=lanczos,setsar=1",
            "-an",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
        shell=False,
    )
    subprocess.run(
        [
            *common,
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2:flags=lanczos,setsar=1",
            "-map_metadata",
            "-1",
            str(crop_path),
        ],
        check=True,
        shell=False,
    )
    html_path = prepared.paths.previews / "index.html"
    html_path.write_text(
        render_preview_html(
            (
                PreviewItem(
                    candidate_id="source-reference",
                    label="Source reference",
                    video_filename=video_name,
                    crop_filenames=(crop_name,),
                    source_path=prepared.selection.canonical_path,
                ),
            )
        ),
        encoding="utf-8",
    )
    return 0, {
        "state": "ready",
        "preview": html_path.name,
        "video": video_name,
        "crop": crop_name,
    }


def _benchmark(args: argparse.Namespace) -> tuple[int, object]:
    started = time.monotonic()
    result = run_phase0_smoke(
        args.output_root / "engvit-benchmark-smoke",
        _binary(args.ffmpeg, "ffmpeg"),
    )
    elapsed = Decimal(str(time.monotonic() - started))
    benchmark = record_benchmark(
        frames=6,
        elapsed_samples=(elapsed,),
        peak_vram_bytes=0,
        peak_disk_bytes=(
            result.artifact.bytes if result.artifact is not None else 0
        ),
        worker_count=1,
    )
    return 0, {
        "state": result.state,
        "scope": "six_frame_plumbing_smoke_only",
        "admissible_for_full_job": False,
        "reason": "full target-geometry benchmark and live resource admission required",
        "benchmark": benchmark,
    }


def _run(args: argparse.Namespace, *, force_resume: bool = False) -> tuple[int, object]:
    prepared = _prepared(args)
    result = run_prepared_phase0(
        prepared,
        max_new_chunks=getattr(args, "max_new_chunks", None),
        resume_paused=force_resume or getattr(args, "resume_paused", False),
    )
    artifact = None
    if result.artifact is not None:
        artifact = {
            "filename": result.artifact.path.name,
            "bytes": result.artifact.bytes,
            "sha256": result.artifact.sha256,
            "frames": result.artifact.frame_count,
        }
    return 0, {
        "state": result.state,
        "completed_chunks": len(result.completed_chunks),
        "total_chunks": len(prepared.chunks),
        "artifact": artifact,
    }


def _pause(args: argparse.Namespace) -> tuple[int, object]:
    prepared = _prepared(args)
    generation = prepared.coordinator.request_pause()
    return 0, generation


def _qa(args: argparse.Namespace) -> tuple[int, object]:
    prepared = _prepared(args)
    artifact = _artifact_from_existing(prepared)
    evidence = run_structural_qa(
        artifact,
        expected_frame_count=len(prepared.timeline.output_frames),
        expected_dimensions=prepared.geometry.target_size,
        expected_time_base=prepared.timeline.output_time_base,
    )
    passed = all(item.state == "PASS" for item in evidence)
    return (0 if passed else 2), {
        "state": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "artifact_sha256": artifact.sha256,
    }


def _persist(args: argparse.Namespace) -> tuple[int, object]:
    paths = create_job_paths(args.output_root, args.job_id)
    receipt = prepare_continuation(paths, args.archive)
    return 0, {
        "archive_filename": receipt.archive_path.name,
        "archive_bytes": receipt.archive_bytes,
        "archive_sha256": receipt.archive_sha256,
        "file_count": len(receipt.files),
        "files": receipt.files,
    }


def _dispatch(args: argparse.Namespace) -> tuple[int, object]:
    handlers: dict[str, Handler] = {
        "preflight": _preflight,
        "discover": _discover,
        "analyze": _analyze,
        "preview": _preview,
        "benchmark": _benchmark,
        "run": _run,
        "pause": _pause,
        "resume": lambda value: _run(value, force_resume=True),
        "qa": _qa,
        "persist": _persist,
    }
    return handlers[args.command](args)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run one CLI action and return a process status for tests/notebooks."""
    try:
        arguments = _parser().parse_args(argv)
        status, payload = _dispatch(arguments)
        _emit(stdout, payload)
        return status
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        stderr.write(
            f"ERROR {type(exc).__name__}: operation failed; "
            "sensitive paths and media names are redacted\n"
        )
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
