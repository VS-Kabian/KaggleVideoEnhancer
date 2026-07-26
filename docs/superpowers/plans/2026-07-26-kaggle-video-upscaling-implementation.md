# Kaggle Video Enhancement and Upscaling Implementation Plan

**Goal:** Implement and release a fail-closed Kaggle notebook/package that analyzes, enhances, upscales, validates, pauses, persists, and resumes private 10-15 minute videos, with measured 4K support and experimental guarded 8K.

**Architecture:** FFmpeg owns media decode/filter/encode/remux; Python owns canonical plans, diagnostics, model inference, tiling, orchestration, evidence, and UX. The MVP normalizes supported inputs to a rational CFR progressive timeline before scenes or chunks. A single coordinator commits immutable segment results. Every full neural run requires a frozen `ExecutionPlan` produced by an exact benchmark.

**Tech stack:** active Kaggle GPU image (currently Python 3.12; probe, do not assume), Python packaging, Pydantic v2, NumPy, OpenCV headless, PyTorch supplied by Kaggle, Spandrel, SafeTensors, FFmpeg/FFprobe, pytest, Hypothesis, Ruff, mypy, Jinja2 with autoescape, ipywidgets, KaggleHub/Kaggle CLI.

**Design source:** [`docs/plans/2026-07-26-kaggle-video-upscaling-research-and-design.md`](../../plans/2026-07-26-kaggle-video-upscaling-research-and-design.md)  
**Validation source:** [`docs/validation/2026-07-26-plan-validation-report.md`](../../validation/2026-07-26-plan-validation-report.md)

## Global Constraints

- Use test-first increments: add a failing test, run it, implement the smallest coherent behavior, then rerun focused and regression tests.
- Never modify Kaggle's Torch/torchvision/CUDA/NumPy stack during a user job.
- Production runtime loads only allowlisted SafeTensors. Pickle-family checkpoints are rejected.
- Full source SHA-256 is mandatory before processing/resume; a sampled fingerprint is only a discovery cache hint.
- HDR is rejected in MVP. There is no implicit tone-map path.
- VFR input is explicitly normalized to CFR. True-VFR output is out of MVP scope.
- Only the coordinator writes manifests. Workers never write shared state.
- No full-frame sequence is written to disk. Queues and caches are byte-bounded.
- 4K is released only after full real-model acceptance on the minimum supported Kaggle GPU. 8K is disabled unless exact per-environment gates pass.
- Missing required QA evidence is `NOT_EVALUATED`, never success.
- All JSON identity material is canonical UTF-8, sorted-key, compact, `allow_nan=false`.
- Commands below are labeled for PowerShell or Kaggle/Linux; do not mix shell syntax.
- Do not mark a task complete until its acceptance evidence is stored under `evidence/`.

---

## Fixed Interfaces and Artifact Contract

Implement these names once; later tasks extend behavior without renaming them.

```python
# src/engvit/types.py
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
MetricState = Literal["PASS", "FAIL", "NOT_EVALUATED"]
HDRPolicy = Literal["reject"]
FPSPolicy = Literal["source_cfr", "normalize_cfr", "rife"]
ContainerPolicy = Literal["mp4_compatibility", "mkv_preservation"]

@dataclass(frozen=True)
class Rational:
    numerator: int
    denominator: int

@dataclass(frozen=True)
class JobPaths:
    root: Path
    artifacts: Path
    segments: Path
    partials: Path
    reports: Path
    previews: Path
    evidence: Path

@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None
    time_base: Rational | None
    start_pts: int | None
    duration_pts: int | None
    disposition: dict[str, bool]
    language: str | None
    metadata: dict[str, str]

@dataclass(frozen=True)
class VideoStreamInfo(StreamInfo):
    coded_width: int
    coded_height: int
    sample_aspect_ratio: Rational | None
    avg_frame_rate: Rational | None
    real_frame_rate: Rational | None
    field_order: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    display_matrix: tuple[float, ...] | None

@dataclass(frozen=True)
class MediaInfo:
    source: Path
    source_sha256: str
    format_name: str
    duration_seconds: Decimal | None
    streams: tuple[StreamInfo, ...]
    selected_video_index: int

@dataclass(frozen=True)
class SourceFrameTiming:
    source_index: int
    best_effort_pts: int
    duration_pts: int
    source_time_base: Rational
    repeat_pict: int
    interlaced: bool
    top_field_first: bool | None

@dataclass(frozen=True)
class OutputFrameSpec:
    output_index: int
    output_pts: int
    output_duration: int
    source_indexes: tuple[int, ...]
    interpolation_fraction: Rational | None

@dataclass(frozen=True)
class TimelinePlan:
    source_time_base: Rational
    output_time_base: Rational
    output_fps: Rational
    timing_transform: tuple[str, ...]
    source_frames: tuple[SourceFrameTiming, ...]
    output_frames: tuple[OutputFrameSpec, ...]
    sha256: str

@dataclass(frozen=True)
class GeometryPlan:
    coded_size: tuple[int, int]
    oriented_size: tuple[int, int]
    target_size: tuple[int, int]
    model_size: tuple[int, int]
    input_sar: Rational | None
    output_sar: Rational
    pixel_transform: tuple[str, ...]
    final_resize: Literal["lanczos"] | None

@dataclass(frozen=True)
class DiagnosticReport:
    source_sha256: str
    timeline_sha256: str
    scan_rows_sha256: str
    sample_indexes: tuple[int, ...]
    features: dict[str, JSONValue]
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class EncoderConfig:
    ffmpeg_sha256: str
    encoder: str
    hardware_device: int | None
    codec: str
    pixel_format: str
    rate_control: dict[str, JSONValue]
    preset: str
    gop: int
    b_frames: int
    output_fps: Rational
    output_time_base: Rational
    color: dict[str, str]
    self_test_sha256: str

@dataclass(frozen=True)
class TilePolicy:
    tile_size: int
    context_pad: int
    blend_overlap: int
    precision: Literal["fp32", "fp16"]
    device_id: int
    calibration_sha256: str

@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    model_id: str
    model_scale: int
    denoise_strength: Decimal | None
    fps_policy: FPSPolicy
    final_resize: Literal["lanczos"] | None

@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    source_decode_start_pts: int
    source_decode_end_pts: int
    output_core_start: int
    output_core_end: int
    context_before: int
    context_after: int
    scene_ids: tuple[int, ...]
    identity_sha256: str

@dataclass(frozen=True)
class ChunkCompletion:
    chunk_id: str
    lease_id: str
    partial_path: Path
    bytes: int
    sha256: str
    frame_count: int
    first_pts: int
    last_pts: int
    boundary_frame_hashes: tuple[str, ...]
    encoder_extradata_sha256: str
    observations: dict[str, JSONValue]

@dataclass(frozen=True)
class BenchmarkResult:
    frames: int
    elapsed_seconds: Decimal
    end_to_end_fps: Decimal
    peak_vram_bytes: int
    peak_disk_bytes: int
    worker_count: int
    variance: Decimal

@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: str
    source_sha256: str
    selection_sha256: str
    timeline_sha256: str
    diagnostic_sha256: str
    recipe: Recipe
    geometry: GeometryPlan
    tiles: tuple[TilePolicy, ...]
    encoder: EncoderConfig
    chunks: tuple[ChunkSpec, ...]
    benchmark: BenchmarkResult
    environment_sha256: str
    required_disk_bytes: int
    predicted_seconds: int
    safety_seconds: int
    identity_sha256: str

@dataclass(frozen=True)
class MetricEvidence:
    evidence_id: str
    protocol: Literal[
        "synthetic_hr_fidelity", "encoder_roundtrip",
        "blind_real_source_consistency", "structural"
    ]
    state: MetricState
    metric: str
    value: JSONValue
    threshold_version: str | None
    inputs: dict[str, str]
    implementation: dict[str, str]
    reason: str | None

class FrameEnhancer(Protocol):
    @property
    def scale(self) -> int: ...
    def enhance(self, frame_rgb: np.ndarray) -> np.ndarray: ...
    def close(self) -> None: ...

class FrameInterpolator(Protocol):
    def interpolate(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray,
        fractions: tuple[Rational, ...],
    ) -> tuple[np.ndarray, ...]: ...
    def close(self) -> None: ...
```

Canonical files:

```text
jobs/<job-id>/
  artifacts/selection.json
  artifacts/environment-lock.json
  artifacts/media-info.json
  artifacts/timeline-plan.json
  artifacts/geometry-plan.json
  artifacts/diagnostic-report.json
  artifacts/recipe.json
  artifacts/execution-plan.json
  artifacts/job-manifest.json
  segments/
  partials/
  reports/qa-report.json
  reports/qa-report.html
  reports/delivery-receipt.json
  previews/
  evidence/
```

`JobManifest` is a versioned Pydantic discriminated union with states:
`created`, `planned`, `running`, `pause_requested`, `paused`, `finalizing`, `qa`, `persisting`, `complete`, `failed`. It contains generation, immutable plan identity, chunk leases/status, artifact hashes, failure records, and persistence receipt. Volatile progress observations live in `progress.json` and are excluded from identity.

Supporting records are strict Pydantic models, not open dictionaries:

| Record | Required fields |
|---|---|
| `AppConfig` | schema version; allowlisted input/weight/wheel/output roots; maximum target dimensions; safety reserves |
| `JobConfig` | selected video index; target box; preset; container; HDR/FPS/stream policies; recipe overrides; experimental consent |
| `EnvironmentLock` | Python/package/Torch/CUDA/GPU/driver/FFmpeg versions, origins, hashes, capabilities, disk roots |
| `KaggleContext` | notebook ID/visibility, internet state, attached Dataset handles/versions/visibility, observable session facts |
| `VerificationReport` / `PreflightResult` | state, evidence IDs, facts, failures, user actions |
| `IntendedUse` / `LicenseEvidence` | private/commercial/redistribution flags; code and weight terms URLs/snapshots/hashes/reviewer/date/notices |
| `DatasetRoot` / `MediaCandidate` / `Selection` | handle, immutable version, root, relative path, size, probe summary/error, selected stream, full hash |
| `ColorDecision` / `FilterGraph` | declared input/output color contract; ordered FFmpeg filters and complete rendered arguments |
| `ProxyScan` / `Scene` / `SampleSet` | timeline/source hashes, feature versions/rows, normalized frame ranges, deterministic sample indexes |
| `ChunkPolicy` / `ChunkLease` / `ManifestGeneration` | normalized target size, context rules, lease ID/worker/expiry, generation/hash |
| `ArtifactReceipt` / `VideoArtifact` / `FinalArtifact` | path relative to job root, bytes, SHA-256, streams, timing, media policy/loss receipt |
| `ModelArtifact` / `SafeTensorSet` / `DeviceSpec` | code/weight/license evidence, architecture signature, tensor inventory, device/capability |
| `TileCalibration` / `TileLayout` / `SeamEvidence` | tested attempts/corpus, exact slices/weights, boundary/interior distributions and state |
| `ReleaseCapabilities` / `RecipeCandidate` / `PreviewResult` / `PreviewSelection` | signed environment/model capability, parameters/cost, proxy/crop receipts, chosen candidate |
| `PlanningContext` / `PlanInputs` / `DiskEstimate` / `LiveResources` / `AdmissionDecision` | bound artifacts; per-phase disk; observed time/disk/VRAM/encoder; pass/refusal/evidence |
| `SyntheticCase` / `FrameSet` / `QualityPolicy` / `QualityDecision` | reference/degradation hashes; aligned frames/PTS; required evidence/threshold version; decision/reasons |
| `TemporalCorpus` / `WindowPolicy` | sequence/cut hashes; context/core sizes; memory/performance/equivalence evidence |
| `PersistenceTarget` / `PersistenceReceipt` | target kind/visibility/handle/version; file allowlist/hashes; remote verification and retrieval |
| `AcceptanceMatrix` / `AcceptanceReport` | exact environments/jobs/gates; immutable evidence references; derived capabilities |

`AppConfig` and `JobConfig` may only use values represented in these schemas. No task may smuggle executable FFmpeg arguments, paths, environment variables, or model URLs through a free-form user dictionary. `JSONValue` maps are permitted only for measured facts or a closed adapter-specific schema that is separately validated.

---

### Task 1: Scaffold the package, schemas, canonical serialization, and fixtures

**Files:**

- Create: `pyproject.toml`, `requirements/{base,dev,kaggle}.txt`
- Create: `src/engvit/{__init__,types,config,canonical,errors,paths}.py`
- Create: `tests/{conftest,test_config,test_canonical,test_paths}.py`
- Create: `tests/fixtures/factory.py`, `scripts/generate_media_fixtures.py`

**Interfaces:**

- Implement the fixed dataclasses/types above and `AppConfig`, `JobConfig`.
- `canonical_bytes(value: object, *, projection: str) -> bytes`
- `canonical_sha256(value: object, *, projection: str) -> str`
- `create_job_paths(root: Path, job_id: str) -> JobPaths`

**Steps:**

- [ ] Configure a `src/` package installed editable in development; no `sys.path` notebook hacks.
- [ ] Make strict Pydantic configs reject unknown keys, invalid rationals, HDR policies other than `reject`, unsafe paths, and unbounded target dimensions.
- [ ] Encode Fraction/Rational, Decimal, Path, enum, bytes hash references, tuples, and dataclasses deterministically; reject NaN/Inf.
- [ ] Define identity projections that exclude timestamps, host paths, ETA, and progress.
- [ ] Generate short deterministic fixtures: CFR 30000/1001, irregular VFR, B-frames, negative/non-zero starts, interlace, five-phase telecine, rotation/flip, non-square SAR, BT.601/709 range variants, PQ/HLG, multiple audio/subtitle streams, chapters/attachments, and malformed probe metadata.
- [ ] Store generation commands and SHA-256; do not commit large binaries.

**Verification:**

```powershell
# Local PowerShell
python -m pip install -e ".[dev]"
python -m pytest tests/test_config.py tests/test_canonical.py tests/test_paths.py -q
python -m ruff check src tests
python -m mypy src
```

**Acceptance:** same semantic object hashes identically across runs/path roots; invalid/volatile data fails or is excluded exactly as documented; fixtures regenerate byte-identically with the pinned FFmpeg image.

---

### Task 2: Implement safe environment, privacy, dependency, and license preflight

**Files:**

- Create: `src/engvit/{environment,privacy,supply_chain,licenses}.py`
- Create: `scripts/{build_wheelhouse,verify_wheelhouse,convert_checkpoint}.py`
- Create: `licenses/{code-components,model-weights,ffmpeg-policy}.yaml`
- Create: `tests/test_{environment,privacy,supply_chain,licenses}.py`

**Interfaces:**

- `capture_environment() -> EnvironmentLock`
- `verify_wheelhouse(root: Path, lock: Path) -> VerificationReport`
- `SensitiveMediaPreflight.run(context: KaggleContext) -> PreflightResult`
- `LicenseRegistry.require_model(model_id: str, intended_use: IntendedUse) -> LicenseEvidence`

**Steps:**

- [ ] Capture runtime/binary/GPU/disk facts listed in the design, including FFmpeg `-L`, `-buildconf`, encoder help, origins, and hashes.
- [ ] Build a binary-only, hash-locked, SBOM-recorded wheelhouse. Reject VCS URLs, sdists, unhashed wheels, runtime internet, and silent resolver changes.
- [ ] Record Torch/torchvision/CUDA/NumPy before and after installation and fail on any replacement.
- [ ] Implement private notebook/Dataset/internet-disabled verification through available Kaggle APIs; if visibility cannot be queried, require and record explicit attestation before media access.
- [ ] Separate repository-code and checkpoint-weight terms. Unverified/prohibited terms block model use.
- [ ] Implement legacy conversion as a separate networkless CPU process with PyTorch 2.6 or newer plus current security patches, `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`, explicit `weights_only=True`, resource limits, no media/secrets, tensor inventory, and SafeTensors output. Runtime rejects `.pth/.pt/.ckpt/.pkl`.
- [ ] Generate `release-capabilities.json`; default `4k=false`, `8k=false`.

**Verification:**

```powershell
python -m pytest tests/test_environment.py tests/test_privacy.py tests/test_supply_chain.py tests/test_licenses.py -q
```

**Acceptance:** malicious path/archive/checkpoint fixtures, visibility unknown, internet enabled, unverified weight terms, Torch drift, and metadata canaries all fail before source frames are read.

---

### Task 3: Select and probe media, hash the source, and freeze orientation/geometry/color

**Files:**

- Create: `src/engvit/media/{selection,probe,geometry,color,path_security}.py`
- Create: `tests/media/test_{selection,probe,geometry,color,path_security}.py`

**Interfaces:**

- `discover_media(roots: tuple[DatasetRoot, ...]) -> tuple[MediaCandidate, ...]`
- `select_media(candidate: MediaCandidate, video_stream_index: int) -> Selection`
- `probe_media(selection: Selection) -> MediaInfo`
- `plan_geometry(video: VideoStreamInfo, config: JobConfig) -> GeometryPlan`
- `classify_color(video: VideoStreamInfo) -> ColorDecision`

**Steps:**

- [ ] Group candidates by Dataset handle/version; exclude weights/wheels/outputs; show size, probe duration, dimensions, stream count, and failure reason.
- [ ] Resolve strict contained regular files; reject symlink/reparse/device/FIFO/socket; re-stat identity while open.
- [ ] Compute full SHA-256 before planning; persist `selection.json` with Dataset handle/version/path and hash.
- [ ] Parse every stream with nullable rationals and full display matrix. Require selected video index.
- [ ] Use `-noautorotate`; plan matrix application once, square-pixel output, `setsar=1`, and stale display-metadata removal.
- [ ] Reject non-orthogonal transforms, encrypted/undecodable sources, PQ/HLG/Dolby Vision/ambiguous HDR, and color metadata insufficient for a safe SDR decision.
- [ ] Plan model-native scale plus at most one named Lanczos final resize.

**Verification:**

```powershell
python -m pytest tests/media/test_selection.py tests/media/test_probe.py tests/media/test_geometry.py tests/media/test_color.py tests/media/test_path_security.py -q
```

**Acceptance:** every fixture yields the expected selected stream, hash, orientation, SAR/DAR, geometry, and color decision; HDR and unsafe paths stop without partial output.

---

### Task 4: Build the complete source timing pass and normalized CFR timeline

**Files:**

- Create: `src/engvit/media/{frame_probe,timeline,filters}.py`
- Create: `tests/media/test_{frame_probe,timeline,filter_chunk_equivalence}.py`

**Interfaces:**

- `stream_source_timing(media: MediaInfo) -> Iterator[SourceFrameTiming]`
- `plan_timeline(media: MediaInfo, config: JobConfig) -> TimelinePlan`
- `build_timing_filter(plan: TimelinePlan, geometry: GeometryPlan) -> FilterGraph`
- `bind_raw_frame(output_index: int, plan: TimelinePlan) -> OutputFrameSpec`

**Steps:**

- [ ] Stream complete `ffprobe -show_frames`; never hold an unbounded JSON response in memory.
- [ ] Reject duplicate/missing/unusable timestamps unless a documented repair policy produces a deterministic mapping.
- [ ] Keep exact rational rates including 30000/1001. Make VFR-to-CFR visible in configuration and report.
- [ ] Make bwdif mode, parity, and output rate explicit.
- [ ] Treat fieldmatch/decimate as continuous. Prove cadence-safe boundaries or render a progressive compressed mezzanine before SR.
- [ ] Apply pre/post-roll for stateful filters and trim only normalized core.
- [ ] Persist full source-to-output mapping. Raw RGB frames carry index only, not fictional PTS.

**Verification:**

```powershell
python -m pytest tests/media/test_frame_probe.py tests/media/test_timeline.py tests/media/test_filter_chunk_equivalence.py -q
```

**Acceptance:** B-frame, negative/non-zero start, irregular VFR, duplicate/missing timestamp, 30000/1001, interlace, every telecine phase, and mixed cadence fixtures match a continuous-filter reference at every boundary.

---

### Task 5: Implement full-duration proxy analysis, scenes, diagnostics, and deterministic sampling

**Files:**

- Create: `src/engvit/analysis/{proxy_scan,scenes,features,sampling,report}.py`
- Create: `tests/analysis/test_{proxy_scan,scenes,features,sampling,report}.py`

**Interfaces:**

- `scan_proxy(media: MediaInfo, timeline: TimelinePlan) -> ProxyScan`
- `detect_scenes(scan: ProxyScan) -> tuple[Scene, ...]`
- `select_samples(scan: ProxyScan, scenes: tuple[Scene, ...], source_sha256: str, count: int) -> tuple[int, ...]`
- `build_diagnostic_report(...) -> DiagnosticReport`

**Steps:**

- [ ] Scan the entire normalized duration at deterministic cadence with SI/TI, luma/chroma, edge, noise, blocking/ringing, banding, motion, flat/dark/highlight, text/line, face, freeze/black/repeat signals.
- [ ] Calibrate scene thresholds on fixtures; preserve both sides of every selected cut.
- [ ] Stratify 48-96 samples across time, scenes, feature extremes, and content classes using source-hash seed.
- [ ] Persist raw rows, feature versions, confidence, warnings, and hashes. Routing consumes evidence but never hides it.

**Verification:**

```powershell
python -m pytest tests/analysis -q
```

**Acceptance:** selection is deterministic, covers each required stratum, does not cross timing coordinates, and reports low confidence rather than fabricating labels.

---

### Task 6: Implement manifests, chunks, atomic coordinator, and fault recovery

**Files:**

- Create: `src/engvit/orchestration/{chunks,manifest,atomic,coordinator,worker}.py`
- Create: `tests/orchestration/test_{chunks,manifest,atomic,coordinator,faults}.py`

**Interfaces:**

- `plan_chunks(timeline: TimelinePlan, scenes: tuple[Scene, ...], policy: ChunkPolicy) -> tuple[ChunkSpec, ...]`
- `AtomicArtifactWriter.write(path: Path, payload: bytes) -> ArtifactReceipt`
- `Coordinator.lease(worker_id: str) -> ChunkLease | None`
- `Coordinator.commit(completion: ChunkCompletion) -> ManifestGeneration`
- `Coordinator.recover() -> RecoveryReport`

**Steps:**

- [ ] Plan normalized output cores plus source decode/context ranges; include timing/state-safe boundaries.
- [ ] Hash all chunk identity inputs. A changed source/plan/model/tile/encoder/environment invalidates reuse.
- [ ] Use unique same-directory partials, file fsync, atomic replace, POSIX directory fsync, and bounded Windows/OneDrive retry with validated backup.
- [ ] Only coordinator changes manifest generation. Workers return immutable completion records over a queue.
- [ ] Recover stale leases, corrupt partials, and orphan renamed segments. Never accept a segment without full hash, frame count, PTS, boundary hashes, and matching identity.
- [ ] Add explicit `pause_requested -> paused` at the next committed segment.

**Verification:**

```powershell
python -m pytest tests/orchestration -q
```

**Acceptance:** kill during write, after fsync, after rename/before commit, simultaneous workers, stale lease, lost worker/GPU, and same-size wrong source all recover or fail without lost updates.

---

### Task 7: Deliver the deterministic streaming Lanczos pipeline and media-semantic QA

**Files:**

- Create: `src/engvit/media/{decode,encode,segments,concat,remux,streams}.py`
- Create: `src/engvit/pipeline.py`
- Create: `tests/media/test_{decode,encode,segments,concat,remux,streams}.py`
- Create: `tests/integration/test_phase0_pipeline.py`

**Interfaces:**

- `decode_normalized_frames(chunk: ChunkSpec, plan: TimelinePlan) -> Iterator[np.ndarray]`
- `encode_segment(frames: Iterable[np.ndarray], chunk: ChunkSpec, config: EncoderConfig) -> ChunkCompletion`
- `validate_and_concat(completions: tuple[ChunkCompletion, ...], config: EncoderConfig) -> VideoArtifact`
- `remux_ancillary(video: VideoArtifact, media: MediaInfo, policy: ContainerPolicy) -> FinalArtifact`

**Steps:**

- [ ] Decoder uses explicit filter graph, `-noautorotate`, `rgb24`, and verifies exact byte/frame counts by normalized index.
- [ ] Encoder rawvideo stdin declares exact size/rational framerate; use and integration-test the Kaggle-compatible `-vsync 0` passthrough and frozen rational `-enc_time_base` contract; freeze GOP/B-frame/color/time-base behavior.
- [ ] Create per-encoder templates with IDR first frame, closed fixed GOP, scene-cut disabled, repeated headers where required; verify active options.
- [ ] Validate stream layout, codec/profile/level/tag/extradata, dimensions/SAR/pixfmt/field/color/rate/time-base before concat; write exact rational durations.
- [ ] Use generated relative concat names, `-safe 1`, `-nostdin`; scan all packets and decode all frames after concat; compare boundary hashes.
- [ ] Define one presentation origin; map selected streams explicitly; preserve relative offsets and dispositions; do not use `-shortest`.
- [ ] Implement MP4 compatibility and MKV preservation policies, with visible per-stream copy/transcode/omit receipts.
- [ ] Strip metadata by default and test canaries.

**Verification:**

```powershell
python -m pytest tests/media tests/integration/test_phase0_pipeline.py -q
```

**Acceptance:** the complete Phase 0 fixture matrix has exact frame count/PTS, no gap/overlap/freeze/black insertion, signed A/V skew within half an output frame, correct orientation/SAR/color, accounted ancillary streams, and no metadata leakage.

---

### Task 8: Implement safe model registry, Spandrel adapter, Real-ESRGAN variants, and DNI

**Files:**

- Create: `src/engvit/models/{registry,weights,spandrel_adapter,realesrgan,dni}.py`
- Create: `models/registry.yaml`
- Create: `tests/models/test_{registry,weights,spandrel_adapter,realesrgan,dni}.py`
- Create: `tests/fixtures/model_corpus/README.md`

**Interfaces:**

- `ModelRegistry.resolve(model_id: str, environment: EnvironmentLock, intended_use: IntendedUse) -> ModelArtifact`
- `load_enhancer(artifact: ModelArtifact, device: DeviceSpec, precision: str) -> FrameEnhancer`
- `merge_dni(general: SafeTensorSet, weak: SafeTensorSet, strength: Decimal) -> SafeTensorSet`

**Steps:**

- [ ] Registry pins code archive/commit, code license, weight terms/evidence, SafeTensors hash/size, architecture signature, scale, channel order, precision support, and required descriptor padding.
- [ ] Validate structural signatures: RRDB features/blocks/growth/scale; compact features/convolutions/activation/scale.
- [ ] Use Spandrel public `ModelLoader`/`ImageModelDescriptor`; enforce one `[1,C,H,W]` RGB `[0,1]` image; no invented normalization.
- [ ] Keep descriptor size padding separate from EngVit context and blend overlap.
- [ ] Clamp/round exactly once and return RGB `uint8`; test odd/tiny/alpha-stripped inputs.
- [ ] DNI canonicalizes wrappers, requires identical keys/shapes/dtypes and compatible non-floats, interpolates FP32 CPU, stores merged hash.
- [ ] Compare all variants and DNI endpoints/midpoint with the official Real-ESRGAN inference path on a rich corpus.
- [ ] Phrase compatibility evidence narrowly; do not claim all BasicSR is unmaintained.

**Verification:**

```powershell
python -m pytest tests/models -q
```

**Acceptance:** only license-approved allowlisted SafeTensors load; official-path parity, pixel contract, structural signature, corrupted/wrong-model rejection, and DNI endpoint tests pass.

---

### Task 9: Implement overlap tiling, OOM calibration, and precision gates

**Files:**

- Create: `src/engvit/tiling/{layout,blend,calibrate,runner}.py`
- Create: `tests/tiling/test_{layout,blend,calibrate,runner}.py`

**Interfaces:**

- `calibrate_tiling(enhancer: FrameEnhancer, corpus: tuple[np.ndarray, ...], device: DeviceSpec) -> TileCalibration`
- `enhance_tiled(frame: np.ndarray, enhancer: FrameEnhancer, policy: TilePolicy) -> np.ndarray`
- `analyze_seams(tiled: np.ndarray, reference: np.ndarray | None, layout: TileLayout) -> SeamEvidence`

**Steps:**

- [ ] Build symmetric tile coverage for odd dimensions and exact native-scale output.
- [ ] Apply descriptor padding, context padding, crop removal, and normalized separable cosine blend without double weighting.
- [ ] Calibrate on bright/dark/detail/flat/text/noise frames; reduce only the failed tile on OOM and record attempts.
- [ ] Make queues byte-bounded. Refuse policies whose working set plus safety reserve exceeds measured free VRAM.
- [ ] Gate FP16 per model/GPU on NaN/Inf, clipping, PSNR/LPIPS/color/seam distributions and visual corpus; default FP32 on failure.
- [ ] Test shifted grid origins, boundary/interior p50/p95/p99, procedural impulses/checkerboards/gradients, and tiled-vs-untiled references.

**Verification:**

```powershell
python -m pytest tests/tiling -q
```

**Acceptance:** exact dimensions/coverage, no zero-weight pixel, deterministic OOM recovery, calibrated minimum tile, and locked seam evidence pass; no hard-coded arbitrary seam threshold.

---

### Task 10: Implement routing, accessible previews, and recipe freezing

**Files:**

- Create: `src/engvit/recipes/{catalog,routing,preview}.py`
- Create: `src/engvit/reporting/templates/preview.html.j2`
- Create: `tests/recipes/test_{catalog,routing,preview}.py`
- Create: `tests/reporting/test_preview_accessibility.py`

**Interfaces:**

- `eligible_recipes(report: DiagnosticReport, capabilities: ReleaseCapabilities) -> tuple[RecipeCandidate, ...]`
- `render_previews(samples: SampleSet, candidates: tuple[RecipeCandidate, ...]) -> PreviewResult`
- `freeze_recipe(selection: PreviewSelection) -> Recipe`

**Steps:**

- [ ] Always include source/downscale reference and Lanczos. Add only licensed, environment-compatible models.
- [ ] Provide Conservative/Balanced/Detail candidates with exact parameters and estimated cost; no hidden face restoration or unimplemented filter.
- [ ] Generate synchronized H.264/AAC browser proxies plus lossless PNG crops. No autoplay.
- [ ] Use keyboard controls, focus order, text/non-color labels, alt text, 200% zoom layout, and screen-reader status.
- [ ] Make candidate labels blindable and order-randomized for formal review.
- [ ] Use Jinja autoescape and generated IDs; never embed source paths/tags.

**Verification:**

```powershell
python -m pytest tests/recipes tests/reporting/test_preview_accessibility.py -q
```

**Acceptance:** same evidence yields same eligible set; ineligible 8K/RIFE/HDR/weights cannot be selected; proxy playback/crops and automated accessibility checks pass.

---

### Task 11: Implement exact benchmark, disk/time admission, encoder admission, and frozen ExecutionPlan

**Files:**

- Create: `src/engvit/planning/{benchmark,disk,session,encoder_admission,execution_plan}.py`
- Create: `tests/planning/test_{benchmark,disk,session,encoder_admission,execution_plan}.py`

**Interfaces:**

- `benchmark_recipe(context: PlanningContext, recipe: Recipe) -> ExecutionPlan`
- `observe_remaining_session_seconds() -> int | None`
- `estimate_peak_disk(plan_inputs: PlanInputs) -> DiskEstimate`
- `admit(plan: ExecutionPlan, live: LiveResources) -> AdmissionDecision`
- `execute_job(paths: JobPaths, plan: ExecutionPlan) -> FinalArtifact`

**Steps:**

- [ ] Benchmark representative real samples through decode/filter/model/tile/resize/color/encode, at exact target geometry and intended worker concurrency.
- [ ] Include startup, concat/remux, required QA, and persistence overhead; compute uncertainty and safety explicitly.
- [ ] Refuse a full neural run when remaining session time is unknown. Recheck time/disk/VRAM before each lease and finalization.
- [ ] Compute peak disk for analysis, processing, finalization, and persistence phases.
- [ ] Run fast encoder smoke plus exact target geometry/pixfmt/config frames. Admit hardware 10-bit HEVC only through P010 and software x265 only through `yuv420p10le`; reject silent bit-depth/pixel-format downgrade.
- [ ] For 8K, require two high-entropy 7680x4320 HEVC Main10 GOPs with final settings, software decode, optional NVDEC decode, exact profile/level/PTS/frame checks; otherwise reject.
- [ ] Calibrate each GPU, benchmark concurrent workers, bind Torch and encoder to device, and include per-chunk tile policy in identity.
- [ ] Freeze all inputs in `execution-plan.json`; `execute_job` rehashes source/plan/environment/live capability. CLI `run` requires a plan and will not silently replan.

**Verification:**

```powershell
python -m pytest tests/planning -q
```

**Acceptance:** time/disk/VRAM/encoder unknown or insufficient yields a clear refusal; mutations after benchmark invalidate the plan; 8K cannot be enabled by a tiny smoke test.

---

### Task 12: Implement calibrated quality protocols and QA reports

**Files:**

- Create: `src/engvit/quality/{protocols,structural,fidelity,encoder_roundtrip,temporal,color,banding,seams,human,report}.py`
- Create: `quality/{metric-contracts,thresholds}.yaml`
- Create: `src/engvit/reporting/templates/qa.html.j2`
- Create: `tests/quality/test_*.py`

**Interfaces:**

- `run_structural_qa(final: FinalArtifact, plan: ExecutionPlan, media: MediaInfo) -> tuple[MetricEvidence, ...]`
- `run_synthetic_hr_fidelity(case: SyntheticCase) -> tuple[MetricEvidence, ...]`
- `run_encoder_roundtrip(pre_encode: FrameSet, final: FinalArtifact) -> tuple[MetricEvidence, ...]`
- `run_blind_consistency(source: MediaInfo, final: FinalArtifact) -> tuple[MetricEvidence, ...]`
- `job_decision(job_evidence: tuple[MetricEvidence, ...], qualified_recipe_evidence: tuple[MetricEvidence, ...], policy: QualityPolicy) -> QualityDecision`
- `release_decision(evidence: tuple[MetricEvidence, ...], policy: QualityPolicy) -> QualityDecision`

**Steps:**

- [ ] Define exact preprocessing/pooling/version contracts: PSNR/SSIM domain/window/border, LPIPS alex v0.1 RGB `[-1,1]`, VMAF model/hash/NEG/PTS/pooling, CAMBI, CIEDE2000 ROIs/distributions, tOF/tLP, blind flow/residual/flat-flicker/freeze checks.
- [ ] Create recorded second-order synthetic degradations and split versioned calibration versus locked holdout corpora.
- [ ] Begin every numeric threshold as `UNSET`; calibrate on development only, freeze hash, then open holdout once.
- [ ] Require structural/timing, synthetic fidelity, encoder roundtrip, tile, and blinded human evidence for release; blind real-source consistency is diagnostic only.
- [ ] Version release evidence by model/weight/recipe/precision/tile/environment/source-domain. A user job links that evidence and runs structural/timing, encoder-roundtrip, color/banding/seam, and full-duration blind diagnostics; it cannot pretend the user source has an HR reference.
- [ ] Formal human claim follows P.910/BT.500 plan with at least 24 retained screened observers; otherwise label evidence pilot/informal.
- [ ] Report `NOT_EVALUATED` on missing required implementation/model/reference/threshold.
- [ ] Verify exact CFR frame count; signed A/V offset at start/middle/end; all PTS/DTS and boundary conditions; color/range and metadata canaries.

**Verification:**

```powershell
python -m pytest tests/quality -q
```

**Acceptance:** deliberately degraded, flickering, hallucinated, color-shifted, banded, seam-marked, desynchronized, repeated/frozen, and truncated fixtures fail the correct protocol; no blind metric or missing threshold can yield release PASS.

---

### Task 13: Add optional RIFE as a cut-safe timing transform

**Files:**

- Create: `src/engvit/interpolation/{base,rife,timeline}.py`
- Create: `dependencies/rife.lock`
- Create: `tests/interpolation/test_{rife,timeline,cuts}.py`

**Interfaces:**

- Implement `FrameInterpolator`.
- `plan_interpolation(timeline: TimelinePlan, scenes: tuple[Scene, ...], target_fps: Rational) -> TimelinePlan`

**Steps:**

- [ ] Pin official code archive/commit, license, weight terms/hash, Python/Torch/CUDA support in separate offline lock.
- [ ] Make failed active-Kaggle-image smoke return feature unavailable without changing the base environment.
- [ ] Define half-open rational output ticks. A cut tick selects the new-scene first frame.
- [ ] Run after deinterlace/CFR source normalization and before SR by default; compare order in preview for poor inputs.
- [ ] Clear pair buffers at cuts; no cross-cut interpolation.
- [ ] Include interpolation in benchmark, disk/time estimate, chunk identity, and required temporal QA.

**Verification:**

```powershell
python -m pytest tests/interpolation -q
```

**Acceptance:** no duplicated/missing output tick, exact duration, zero cross-cut generated frames, and environment failure leaves the Phase 1 spatial pipeline unchanged.

---

### Task 14: Build temporal challenger harness without coupling it to release

**Files:**

- Create: `src/engvit/models/temporal/{base,windows,realbasicvsr,nanovsr,realviformer}.py`
- Create: `dependencies/temporal/*.lock`
- Create: `tests/models/temporal/test_{windows,adapters,cut_reset}.py`
- Create: `research/model-watchlist.yaml`

**Interfaces:**

- `TemporalEnhancer.enhance_window(frames_rgb: tuple[np.ndarray, ...], core: slice) -> tuple[np.ndarray, ...]`
- `calibrate_temporal_window(model: TemporalEnhancer, corpus: TemporalCorpus, device: DeviceSpec) -> WindowPolicy`

**Steps:**

- [ ] Give each model a separate environment/code/weight/license lock and capability record.
- [ ] Implement bounded window/context/core semantics and explicit cut reset. Do not call a finite bidirectional tensor “persistent hidden state.”
- [ ] Treat RealBasicVSR upstream whole-video/non-overlap logic as reference only; prove custom bounded equivalence.
- [ ] Evaluate NanoVSR and RealViFormer next; record training-degradation limitations.
- [ ] Keep DOVE, Stream-DiffVSR, LiteVSR, InfVSR, and FlashVSR as watchlist entries until reproducible exact hardware/memory/license/code gates pass.
- [ ] Promote only if locked quality improves over Phase 1 and full-duration Kaggle admission remains valid.

**Verification:**

```powershell
python -m pytest tests/models/temporal -q
```

**Acceptance:** arbitrary chunk/window boundaries match continuous reference within calibrated tolerance, no state/context crosses cuts, and no research model appears in the user recipe catalog without a signed capability record.

---

### Task 15: Build the Kaggle notebook, CLI, progress, pause/resume, and output retrieval

**Files:**

- Create: `notebooks/engvit_kaggle.ipynb`
- Create: `src/engvit/{cli,progress}.py`
- Create: `src/engvit/ui/{preflight,selection,analysis,configure,preview,benchmark,run,qa}.py`
- Create: `tests/ui/test_*.py`, `tests/test_cli.py`

**Interfaces:**

- CLI: `engvit preflight|discover|analyze|preview|benchmark|run|pause|resume|qa|persist`
- `ProgressSnapshot(phase, frames, duration, chunks, elapsed, eta_low, eta_high, confidence, disk, vram, last_checkpoint, retry, expected_session_finish)`
- `prepare_continuation(job: JobPaths, destination: PersistenceTarget) -> PersistenceReceipt`
- `resume_continuation(receipt: PersistenceReceipt, attached_root: Path) -> JobPaths`

**Steps:**

- [ ] Implement the eight-section product flow in order; each cell is idempotent or refuses unsafe repetition.
- [ ] Add “Bring your video to Kaggle,” private visibility, attach/rescan, and empty/error states.
- [ ] Default compatible MP4, Balanced, up to 4K, original CFR; advanced hides codec, denoise, stream, disk, and experimental controls.
- [ ] Show phase, frames/time/chunks, ETA range/confidence, disk/VRAM, checkpoint, retry, and expected session finish.
- [ ] Add “Stop after segment,” safe interrupt recovery, “Prepare continuation,” “Resume continuation,” and “Run next slice.”
- [ ] Artifact tiers: Delivery (final, manifest, report, compact previews), Continuation (manifest/segments until verified), Diagnostics (separate opt-in compressed).
- [ ] Persist to a private versioned Dataset/notebook output or generate explicit local download. Record bytes, SHA-256, remote handle/version, and retrieval commands (`kaggle kernels output`/KaggleHub where applicable).
- [ ] Recheck visibility and remote hashes before cleanup. Never depend on `/kaggle/working` across sessions.
- [ ] Clear source names/paths from logs and cell output.

**Verification:**

```powershell
python -m pytest tests/ui tests/test_cli.py -q
jupyter nbconvert --execute notebooks/engvit_kaggle.ipynb --to notebook --output C:\tmp\engvit-smoke.ipynb
```

```bash
# Kaggle/Linux smoke
python -m engvit.cli preflight --config configs/kaggle-smoke.yaml
python -m engvit.cli analyze --selection jobs/smoke/artifacts/selection.json
python -m engvit.cli benchmark --job jobs/smoke
```

**Acceptance:** a novice can select, preview, admit, pause, persist, reattach, resume, verify, and retrieve a result without editing Python; accessibility and privacy tests pass.

---

### Task 16: Execute release matrix, freeze evidence, and enable only proven capabilities

**Files:**

- Create: `acceptance/{matrix,minimum-gpu,4k-real-model,8k-experimental}.yaml`
- Create: `scripts/run_acceptance.py`
- Create: `evidence/releases/<version>/`
- Create: `docs/{operations,privacy,licenses,quality-method,limitations,kaggle-runbook}.md`
- Update: `release-capabilities.json`

**Interfaces:**

- `run_acceptance(matrix: AcceptanceMatrix, environment: EnvironmentLock) -> AcceptanceReport`
- `promote_capability(report: AcceptanceReport, capability: str) -> ReleaseCapabilities`

**Steps:**

- [ ] Freeze source/fixture/model/wheel/code/config/threshold/FFmpeg hashes before runs.
- [ ] Run Phase 0 fixtures and a 15-minute deterministic plumbing job.
- [ ] Attempt the minimum baseline on a P100 first. If it fails or is unavailable for the recorded matrix, scope the public capability to the exact faster GPU class that passes; never market the result as generic Kaggle support.
- [ ] On that explicitly declared minimum supported Kaggle GPU, run **real** end-to-end neural jobs:
  - 15-minute 1920x1080 30 fps compressed/high-motion SDR -> 3840x2160 using approved x2 baseline;
  - 15-minute 1280x720 30 fps mixed-content SDR -> 3840x2160 using approved compact x4 baseline.
- [ ] Each job includes full probe/timeline/analysis, model inference, tiles, final resize, encode/remux, required QA, interruption after a segment, private persistence, new-session hash verification, resume, and retrieval.
- [ ] Record wall time, end-to-end fps, variance, peak VRAM/disk, encoder path, artifact hashes, all QA evidence, and operator steps. A fake enhancer or short neural clip cannot satisfy this gate.
- [ ] Run dual-GPU acceptance only if exposed; compare measured concurrent scaling and contention.
- [ ] Run 8K only as a separate experimental matrix: exact 7680x4320 HEVC Main10 encoder/decode admission, full recipe benchmark, storage/session plan, interruption/persistence/resume, and complete QA.
- [ ] Enable `4k` only if both required jobs pass. Enable `8k` only for the exact environment/recipe signature that passes. Otherwise publish the refusal reason.
- [ ] Perform an independent source-to-evidence audit; no author self-attestation alone.

**Verification:**

```bash
# Kaggle/Linux release run
python scripts/run_acceptance.py --matrix acceptance/matrix.yaml \
  --evidence-root evidence/releases/<version>
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src
```

**Acceptance:** evidence directory contains immutable receipts for every gate; `release-capabilities.json` is mechanically derived; 4K and 8K remain false on any required failure or `NOT_EVALUATED`.

---

## Dependency Order

```text
1 schemas/fixtures
  -> 2 environment/privacy/licenses
  -> 3 media/orientation/color
  -> 4 timeline
  -> 5 diagnostics
  -> 6 coordinator/chunks
  -> 7 deterministic Phase 0 media pipeline
  -> 8 safe spatial models
  -> 9 tiling
  -> 10 routing/previews
  -> 11 exact benchmark/ExecutionPlan
  -> 12 calibrated QA
  -> 15 notebook/CLI/persistence
  -> 16 real Kaggle acceptance

13 RIFE depends on 4, 7, 11, 12 and must not block Phase 1.
14 temporal challengers depend on 4, 8, 9, 11, 12 and remain isolated.
```

## Cross-Task Release Checklist

- [ ] All fixed types exist and strict serialization/hash tests pass.
- [ ] Full source hash, source selection, environment, timeline, recipe, encoder, and execution plan are mutually bound.
- [ ] Rawvideo index semantics, CFR normalization, filter-aware chunks, and presentation origin pass adversarial fixtures.
- [ ] Coordinator-only manifest mutation and every crash point recover correctly.
- [ ] SafeTensors/license/dependency/privacy/metadata policies fail closed.
- [ ] Real-ESRGAN official parity, tiles, FP16 policy, and exact target benchmark pass.
- [ ] Structural, synthetic, encoder, temporal, color, banding, seam, and human protocols have complete evidence.
- [ ] Notebook accessibility, pause, persistence, resume, and retrieval pass.
- [ ] Two full-duration real-model 4K jobs pass on the declared minimum GPU.
- [ ] 8K remains disabled unless its separate exact matrix passes.

## Final Handoff Rule

Do not call the implementation production-ready because unit tests, a small encoder probe, a fake enhancer, or a short neural sample passes. The release claim is earned only when Task 16 produces immutable full-duration evidence and capability flags derived from it. Until then, report the exact completed phase and the first failed or unevaluated gate.
