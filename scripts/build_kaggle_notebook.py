"""Generate the reader-facing Kaggle notebook with nbformat."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _apply_profile(
    notebook: nbformat.NotebookNode,
    profile: str,
) -> nbformat.NotebookNode:
    if profile == "generic":
        return notebook
    if profile != "mukikabi006":
        raise ValueError(f"unknown notebook profile: {profile}")

    replacements = {
        "# EngVit: private, resumable video enhancement on Kaggle": (
            "# EngVit: mukikabi006 private video enhancement"
        ),
        "    WORKSPACE,\n    Path(\"/kaggle/input/engvit-code\"),": (
            "    Path(\"/kaggle/input/datasets/mukikabi006/engvit-code\"),\n"
            "    WORKSPACE,\n"
            "    Path(\"/kaggle/input/engvit-code\"),"
        ),
        'NESTED_SOURCE_ROOTS = (\n    Path("/kaggle/input/engvit-code"),': (
            "NESTED_SOURCE_ROOTS = (\n"
            '    Path("/kaggle/input/datasets/mukikabi006/engvit-code"),\n'
            '    Path("/kaggle/input/engvit-code"),'
        ),
        "SMOKE_MODE = True": "SMOKE_MODE = False",
        "RUN_JOB = SMOKE_MODE  # changing SMOKE_MODE to False also makes execution opt-in": (
            "RUN_JOB = True  # process one verified segment, then pause"
        ),
        "MAX_NEW_CHUNKS = None  # e.g. 1 stops after the next committed segment": (
            "MAX_NEW_CHUNKS = 1  # stop after the next committed segment"
        ),
        'NOTEBOOK_ID = "owner/engvit-private"': (
            'NOTEBOOK_ID = "mukikabi006/engvit-private"'
        ),
        'NOTEBOOK_VISIBILITY = "unknown"  # private, public, or unknown': (
            'NOTEBOOK_VISIBILITY = "private"  # verify in Kaggle before running'
        ),
        'DATASET_VISIBILITY = "unknown"   # private, public, or unknown': (
            'DATASET_VISIBILITY = "private"   # verify in Kaggle before running'
        ),
        'DATASET_HANDLE = "owner/private-video-dataset"': (
            'DATASET_HANDLE = "mukikabi006/private-video-dataset"'
        ),
        'DATASET_ROOT = Path("/kaggle/input/private-video-dataset")': (
            'DATASET_ROOT = Path('
            '"/kaggle/input/datasets/mukikabi006/private-video-dataset"'
            ")"
        ),
        'RELATIVE_VIDEO_PATH = "video.mp4"': (
            'RELATIVE_VIDEO_PATH = "GH011828.realesrgan.mkv"'
        ),
        'JOB_ID = "engvit-job"': 'JOB_ID = "gh011828-phase0"',
    }
    for cell in notebook.cells:
        source = cell.source
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell.source = source
    notebook.metadata["engvit_profile"] = profile
    return notebook


def build_notebook(
    *,
    profile: str = "generic",
) -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# EngVit: private, resumable video enhancement on Kaggle

## Goal

This tutorial runs EngVit's deterministic **Phase 0 Lanczos baseline** from a
private attached Kaggle Dataset. It probes every stream/frame timestamp, rejects
unsafe HDR/paths, normalizes supported timing, renders verified chunks, and
assembles a delivery artifact.

Neural Real-ESRGAN and 4K/8K release claims remain disabled until the attached
weights, active Kaggle image, full-duration benchmark, and acceptance evidence
pass. The notebook never downloads or executes a checkpoint from the Internet."""
        ),
        new_markdown_cell(
            """## 1. Setup

Run top-to-bottom. `SMOKE_MODE=True` is the safe self-test and requires no user
media. For a real private Dataset, change it to `False`, fill Section 2, confirm
Internet is disabled in Notebook settings, and make the private-media attestation."""
        ),
        new_code_cell(
            """from pathlib import Path
import json
import os
import shutil
import sys

WORKSPACE = Path.cwd().resolve()
SOURCE_ROOT_CANDIDATES = (
    WORKSPACE,
    Path("/kaggle/input/engvit-code"),
    Path("/kaggle/working/EngVit"),
)
NESTED_SOURCE_ROOTS = (
    Path("/kaggle/input/engvit-code"),
)
if Path("/kaggle/input").is_dir():
    SOURCE_ROOT_CANDIDATES += tuple(
        path
        for path in sorted(Path("/kaggle/input").iterdir())
        if path.is_dir()
    )

source_matches = {
    root.resolve()
    for root in SOURCE_ROOT_CANDIDATES
    if (root / "src" / "engvit").is_dir()
}
for attached_root in NESTED_SOURCE_ROOTS:
    if not attached_root.is_dir():
        continue
    contained_root = attached_root.resolve()
    for pattern in ("*/src/engvit", "*/*/src/engvit"):
        for package_path in attached_root.glob(pattern):
            if package_path.is_symlink() or not package_path.is_dir():
                continue
            candidate_root = package_path.parents[1].resolve()
            if candidate_root.is_relative_to(contained_root):
                source_matches.add(candidate_root)

if not source_matches:
    raise RuntimeError(
        "The attached EngVit code Dataset does not expose src/engvit at its "
        "root or beneath one supported wrapper directory."
    )
if len(source_matches) > 1:
    raise RuntimeError("The code Dataset contains ambiguous EngVit source roots.")
SOURCE_ROOT = next(iter(source_matches))
sys.path.insert(0, str(SOURCE_ROOT / "src"))

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("FFmpeg is required and was not found.") from exc
FFMPEG = Path(FFMPEG).resolve()
FFPROBE = shutil.which("ffprobe")
print({"setup": "ready", "python": sys.version.split()[0], "ffprobe": bool(FFPROBE)})"""
        ),
        new_markdown_cell(
            """## 2. Private input and job parameters

### Key assumptions

- The media Dataset is private or inaccessible to untrusted users.
- Kaggle Notebook Internet is **Off** for sensitive media.
- The input is at most 15 minutes and has explicit supported SDR color metadata.
- `relative_video_path` is relative to the attached Dataset root; never paste a
  secret URL or token here.
- Phase 0 uses software H.264 and Lanczos. A target request is not a release
  guarantee; resource admission/full-duration evidence is still required."""
        ),
        new_code_cell(
            """SMOKE_MODE = True
RUN_JOB = SMOKE_MODE  # changing SMOKE_MODE to False also makes execution opt-in
MAX_NEW_CHUNKS = None  # e.g. 1 stops after the next committed segment
RESUME_PAUSED = False  # set True only after verifying the attached continuation
PREPARE_CONTINUATION = False

# Real-job fields (used only when SMOKE_MODE=False)
NOTEBOOK_ID = "owner/engvit-private"
NOTEBOOK_VISIBILITY = "unknown"  # private, public, or unknown
DATASET_VISIBILITY = "unknown"   # private, public, or unknown
VISIBILITY_ATTESTATION = None    # non-empty only after manually verifying unknown states
INTERNET_ENABLED = False
DATASET_HANDLE = "owner/private-video-dataset"
DATASET_VERSION = "1"
DATASET_ROOT = Path("/kaggle/input/private-video-dataset")
RELATIVE_VIDEO_PATH = "video.mp4"
SELECTED_VIDEO_INDEX = None  # None selects the first probed video stream
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160
FPS_POLICY = "source_cfr"  # source_cfr or normalize_cfr
TARGET_FPS_NUMERATOR = None    # e.g. 30 for normalize_cfr
TARGET_FPS_DENOMINATOR = None  # e.g. 1 for normalize_cfr
CHUNK_FRAMES = 300
JOB_ID = "engvit-job"
OUTPUT_ROOT = (
    Path("/kaggle/working/engvit-jobs")
    if Path("/kaggle").exists()
    else WORKSPACE / ".notebook-run"
)

print({
    "mode": "smoke" if SMOKE_MODE else "private_dataset",
    "run_requested": RUN_JOB,
    "target": [64, 36] if SMOKE_MODE else [TARGET_WIDTH, TARGET_HEIGHT],
})"""
        ),
        new_markdown_cell(
            """## 3. Preflight

This gate stops public/unknown visibility, Internet-enabled sensitive runs, and
missing FFprobe for real media. It does not upload, mutate, or decode the source."""
        ),
        new_code_cell(
            """from engvit.privacy import DatasetVisibility, KaggleContext, SensitiveMediaPreflight

if SMOKE_MODE:
    preflight_state = {"passed": True, "mode": "generated_nonsensitive_fixture"}
else:
    if FFPROBE is None:
        raise RuntimeError("Real media preparation requires ffprobe in the Kaggle image.")
    preflight = SensitiveMediaPreflight().run(
        KaggleContext(
            notebook_id=NOTEBOOK_ID,
            notebook_visibility=NOTEBOOK_VISIBILITY,
            internet_enabled=INTERNET_ENABLED,
            datasets=(
                DatasetVisibility(
                    handle=DATASET_HANDLE,
                    version=DATASET_VERSION,
                    role="media",
                    visibility=DATASET_VISIBILITY,
                ),
            ),
            visibility_attestation=VISIBILITY_ATTESTATION,
        )
    )
    if not preflight.passed:
        raise RuntimeError("Sensitive-media preflight failed: " + "; ".join(preflight.failures))
    preflight_state = {"passed": True, "mode": "private_dataset_attested"}
print(preflight_state)"""
        ),
        new_markdown_cell(
            """## 4. Select, probe, and freeze the plan

Real mode performs contained file discovery, complete source SHA-256, FFprobe
stream parsing, full frame-timing scan, SDR color gate, geometry planning, and
resumable chunk identity. It writes immutable evidence under the job directory."""
        ),
        new_code_cell(
            """prepared = None
if not SMOKE_MODE:
    from engvit.kaggle import KagglePhase0Request, prepare_phase0
    from engvit.types import Rational

    if (TARGET_FPS_NUMERATOR is None) != (TARGET_FPS_DENOMINATOR is None):
        raise ValueError("Set both target FPS numerator and denominator, or neither.")
    target_fps = (
        Rational(TARGET_FPS_NUMERATOR, TARGET_FPS_DENOMINATOR)
        if TARGET_FPS_NUMERATOR is not None
        and TARGET_FPS_DENOMINATOR is not None
        else None
    )
    request = KagglePhase0Request(
        dataset_handle=DATASET_HANDLE,
        dataset_version=DATASET_VERSION,
        dataset_root=DATASET_ROOT,
        relative_video_path=RELATIVE_VIDEO_PATH,
        output_root=OUTPUT_ROOT,
        job_id=JOB_ID,
        selected_video_index=SELECTED_VIDEO_INDEX,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        fps_policy=FPS_POLICY,
        target_fps=target_fps,
        chunk_frames=CHUNK_FRAMES,
    )
    prepared = prepare_phase0(
        request,
        ffmpeg_path=FFMPEG,
        ffprobe_path=Path(FFPROBE),
    )
    plan_summary = {
        "source_sha256_prefix": prepared.selection.source_sha256[:12],
        "source_frames": len(prepared.timeline.source_frames),
        "output_frames": len(prepared.timeline.output_frames),
        "output_fps": (
            f"{prepared.timeline.output_fps.numerator}/"
            f"{prepared.timeline.output_fps.denominator}"
        ),
        "target_size": list(prepared.geometry.target_size),
        "chunks": len(prepared.chunks),
        "color_state": prepared.color.state,
    }
else:
    plan_summary = {"mode": "smoke", "planned_frames": 6, "planned_chunks": 2}
print(plan_summary)"""
        ),
        new_markdown_cell(
            """## 5. Analysis and recipe

Phase 0 deliberately freezes one transparent recipe: normalized source timing,
single-pass geometry/color handling, Lanczos spatial resize, and fixed closed-GOP
software H.264. Neural recipes stay unavailable while their registry entries or
release capabilities are unverified."""
        ),
        new_code_cell(
            """capabilities_path = SOURCE_ROOT / "release-capabilities.json"
capabilities = json.loads(capabilities_path.read_text(encoding="utf-8"))
recipe_state = {
    "selected_recipe": "phase0-lanczos-v1",
    "4k_release_enabled": capabilities["capabilities"]["4k"],
    "8k_release_enabled": capabilities["capabilities"]["8k"],
    "rife_enabled": capabilities["capabilities"]["rife"],
    "reason": capabilities["reason"],
}
print(recipe_state)"""
        ),
        new_markdown_cell(
            """## 6. Run or resume at verified chunk boundaries

Each segment is decoded, encoded, decoded again through FFmpeg `framehash`, and
committed by the single manifest coordinator. Re-running this cell recovers the
manifest and reuses only completions whose identity, size, SHA-256, frame count,
PTS, and boundary hashes still match."""
        ),
        new_code_cell(
            """result = None
if RUN_JOB:
    if SMOKE_MODE:
        from engvit.smoke import run_phase0_smoke
        result = run_phase0_smoke(OUTPUT_ROOT / "smoke-runtime", FFMPEG)
    else:
        from engvit.kaggle import run_prepared_phase0
        if prepared is None:
            raise RuntimeError("Run Section 4 before the job.")
        result = run_prepared_phase0(
            prepared,
            max_new_chunks=MAX_NEW_CHUNKS,
            resume_paused=RESUME_PAUSED,
        )
    print({
        "state": result.state,
        "completed_chunks": len(result.completed_chunks),
        "artifact_ready": result.artifact is not None,
    })
else:
    print({"state": "not_started", "instruction": "Set RUN_JOB=True when ready."})"""
        ),
        new_markdown_cell(
            """## 7. Structural checks and output retrieval

The final video is fully decoded for frame count, PTS, duration, and boundary
hash evidence. Kaggle outputs live under `/kaggle/working`; download them before
the session ends or persist a private continuation Dataset. Never assume working
storage survives a new session."""
        ),
        new_code_cell(
            """qa_summary = {"state": "NOT_EVALUATED", "reason": "job_not_run"}
if result is not None and result.artifact is not None:
    from engvit.quality.structural import run_structural_qa
    from engvit.types import Rational

    if SMOKE_MODE:
        expected_frames = 6
        expected_dimensions = (64, 36)
        expected_time_base = Rational(1, 10)
    else:
        if prepared is None:
            raise RuntimeError("Prepared job evidence is unavailable.")
        expected_frames = len(prepared.timeline.output_frames)
        expected_dimensions = prepared.geometry.target_size
        expected_time_base = prepared.timeline.output_time_base
    evidence = run_structural_qa(
        result.artifact,
        expected_frame_count=expected_frames,
        expected_dimensions=expected_dimensions,
        expected_time_base=expected_time_base,
    )
    passed = all(item.state == "PASS" for item in evidence)
    qa_summary = {
        "state": "PASS" if passed else "FAIL",
        "metrics": {item.metric: item.state for item in evidence},
        "frames": result.artifact.frame_count,
        "first_pts": result.artifact.first_pts,
        "last_pts": result.artifact.last_pts,
        "time_base": (
            f"{result.artifact.time_base.numerator}/"
            f"{result.artifact.time_base.denominator}"
        ),
        "artifact_bytes": result.artifact.bytes,
        "artifact_sha256": result.artifact.sha256,
        "output_filename": result.artifact.path.name,
    }
print(qa_summary)"""
        ),
        new_code_cell(
            """continuation_summary = {"state": "not_requested"}
if PREPARE_CONTINUATION:
    if SMOKE_MODE:
        raise RuntimeError("Continuation packaging is for real private jobs.")
    if prepared is None or result is None:
        raise RuntimeError("Prepare and run a segment before persistence.")
    from engvit.persistence import prepare_continuation
    receipt = prepare_continuation(
        prepared.paths,
        OUTPUT_ROOT / f"{JOB_ID}-private-continuation.zip",
    )
    continuation_summary = {
        "state": "ready",
        "archive_bytes": receipt.archive_bytes,
        "archive_sha256": receipt.archive_sha256,
        "archive_path": str(receipt.archive_path),
        "privacy": "Store only in a private Dataset or private local destination.",
    }
print(continuation_summary)"""
        ),
        new_markdown_cell(
            """## 8. Checks and next steps

- A smoke `PASS` proves notebook plumbing on this environment, not 10-15 minute
  4K/8K neural capability.
- For a real source, review `selection.json`, `timeline.json`, `geometry.json`,
  `environment.json`, `manifest.json`, and the final hash before retrieval.
- If the job is paused or Kaggle time is insufficient, preserve the entire job
  directory as a **private** continuation artifact and verify hashes after reattach.
- Real-ESRGAN becomes selectable only after exact approved SafeTensors, Spandrel
  parity, tile/precision calibration, resource admission, and full-duration
  acceptance evidence are attached. No cell silently enables it."""
        ),
    ]
    for index, cell in enumerate(cells):
        digest = hashlib.sha256(cell.source.encode("utf-8")).hexdigest()[:12]
        cell["id"] = f"engvit-{index:02d}-{digest}"

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    return _apply_profile(notebook, profile)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("generic", "mukikabi006"),
        default="generic",
    )
    args = parser.parse_args()
    notebook = build_notebook(profile=args.profile)
    nbformat.validate(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
