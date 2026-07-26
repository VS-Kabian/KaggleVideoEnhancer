# Kaggle Video Enhancement and Upscaling: Validated Research and Technical Design

**Date:** 2026-07-26  
**Status:** Ready for implementation; runtime release gates not yet executed  
**Target:** Private, user-supplied 10-15 minute videos; reliable bounded 4K, experimental guarded 8K  
**Companion documents:** [validation report](../validation/2026-07-26-plan-validation-report.md) and [implementation plan](../superpowers/plans/2026-07-26-kaggle-video-upscaling-implementation.md)

## 1. Decision

Build a resumable, scene-aware Kaggle pipeline around FFmpeg and a safe Real-ESRGAN-family spatial baseline. Analyze the whole source cheaply, let the user compare bounded previews, benchmark the exact selected recipe on the active GPU, and admit a full job only when measured time, VRAM, disk, encoder, privacy, weight-license, and persistence gates pass.

The first release targets **up to 3840x2160**. It does not promise that neural upscaling reconstructs ground-truth detail; it creates a perceptually clearer rendition while measuring artifacts and preserving media semantics. **7680x4320 stays disabled by default** and is exposed only as an experimental per-job option after exact 8K admission.

The release baseline is:

- deterministic Lanczos for plumbing, comparison, and fallback;
- Real-ESRGAN x2plus for common 1080p to 4K work;
- Real-ESRGAN x4plus, anime x4plus, and compact general x4v3 as preview challengers;
- RIFE only as optional frame-rate conversion, never described as super-resolution;
- temporal/generative VSR behind later, model-specific gates.

## 2. Claims and non-claims

### Supported outcome after release gates pass

- accept one selected, private video from a Kaggle Dataset;
- support 10-15 minute SDR inputs at the tested resolutions/codecs;
- inspect all streams and the complete source-frame timeline;
- normalize supported input to an explicit CFR progressive timeline;
- diagnose compression, blur, noise, banding, ringing, motion, text/line structure, faces, animation, interlace/telecine, and scene cuts;
- compare deterministic and neural candidates on representative samples;
- upscale with bounded memory using overlap tiles and resumable compressed segments;
- preserve selected compatible ancillary streams or report every conversion/loss;
- produce a compatible final video, manifest, QA report, compact previews, and SHA-256 receipt;
- stop safely after a segment and resume from a verified private continuation bundle.

### Explicit non-claims

- It cannot recover unknowable detail or certify factual fidelity.
- It does not decrypt or circumvent protected media.
- It does not support HDR in the MVP; PQ/HLG and ambiguous HDR fail closed.
- It does not preserve source VFR as VFR; supported VFR is explicitly normalized to a selected CFR.
- It does not promise a stable Kaggle GPU, quota, session duration, disk allocation, or hardware encoder.
- It does not call 8K generally available.
- It does not claim scientific perceptual superiority from an unblinded preview or an uncalibrated metric.

## 3. Current platform facts and reproducibility

Kaggle resources are volatile. Official documentation describes demand-dependent accelerator availability and quota behavior rather than a permanent contract. The current Kaggle GPU image must be inspected at runtime; the active upstream image is Python 3.12 at this design date, so a fixed Python 3.10/3.11 assumption is invalid.

Every job records:

- Kaggle image identifier where observable;
- Python, FFmpeg, FFprobe, NVIDIA driver, CUDA runtime, Torch, torchvision, NumPy, and package versions;
- GPU name/count, compute capability, total/free VRAM;
- disk totals/free space for input, working, and output roots;
- `ffmpeg -version`, `-L`, `-buildconf`, encoder help, and binary SHA-256/origin;
- wheelhouse manifest/SBOM and every model/code/weight hash;
- notebook, package, configuration, recipe, timeline, encoder, and execution-plan hashes.

The install path is offline-first:

1. build a binary-only wheelhouse outside the production run;
2. lock every artifact with a SHA-256 hash and SBOM;
3. attach it as a private/versioned Kaggle Dataset;
4. preflight the installed Kaggle Torch/CUDA stack;
5. install with `--no-index --find-links ... --require-hashes`, using `--no-deps` only for audited packages whose requirements are prevalidated;
6. compare Torch, torchvision, CUDA, and NumPy before and after; any replacement fails.

No runtime Git clone, VCS URL, source distribution, unpinned download, or internet-dependent model fetch is allowed.

## 4. Product flow

The notebook is a linear state machine:

`Preflight -> Select -> Analyze -> Configure -> Preview -> Benchmark -> Run/Resume -> QA/Persist`

1. **Preflight:** confirm private visibility or obtain a recorded attestation, internet-disabled mode, safe environment, wheel/weight licenses, disk, GPU, and basic encode/decode.
2. **Select:** enumerate attached Datasets by handle/version, filter media, exclude model assets, and persist the exact selection.
3. **Analyze:** probe streams, hash the full source, build the timeline, generate a full-duration proxy scan, scenes, diagnostics, and deterministic samples.
4. **Configure:** default to compatible MP4, Balanced, up to 4K, original frame rate; advanced settings expose codecs, denoise, stream policy, and disk tradeoffs.
5. **Preview:** render synchronized H.264/AAC proxies and PNG crops for Lanczos and eligible neural recipes.
6. **Benchmark:** test the exact output geometry, encoder configuration, model, tiles, precision, and concurrent worker count; freeze an `ExecutionPlan`.
7. **Run/Resume:** process segments with progress, ETA range/confidence, disk/VRAM, pause-after-segment, retry, and continuation controls.
8. **QA/Persist:** verify structure and required evidence, create allowlisted delivery/continuation bundles, persist or download, rehash, then permit cleanup.

Failure cards say what failed, what remains valid, the safe retry, the user action, and optional technical detail. Browser previews never assume 4K/8K codec support.

## 5. Media and timeline contract

### 5.1 Complete probe

Use `ffprobe` JSON for:

- container format, duration, start time, bitrate, chapters, attachments, programs, and tags;
- every video/audio/subtitle/data stream, codec, profile, level, dimensions, sample/display aspect, field order, frame rates, time base, start/duration timestamps, disposition, language, and color fields;
- complete display matrix, including rotation/reflection;
- a streamed **complete** selected-video `-show_frames` timing pass recording frame index, best-effort timestamp, packet duration, repeat flags, and interlace flags.

Malformed or missing rational fields remain nullable; no divide-by-zero or fabricated rate. The user explicitly selects a video stream if more than one is present.

Before reading media, the path is resolved strictly within an allowlisted input root, must be a regular file, and is rejected if it is a symlink, reparse point, device, FIFO, or socket. The file identity is restatted while open. A sampled fingerprint is only a cache hint; full SHA-256 is mandatory before neural processing or resume.

### 5.2 Orientation and geometry

FFmpeg runs with `-noautorotate`. EngVit parses and applies the complete display matrix exactly once, including orthogonal flips. Non-orthogonal transforms are rejected in MVP. Anamorphic sources are resampled to square pixels; output uses `setsar=1`, and stale rotation/display metadata is removed.

For target box `(Wmax,Hmax)`, oriented source `(Ws,Hs)`, model scale `s`, and even-dimension quantum `q=2`:

```text
r = min(Wmax / Ws, Hmax / Hs)
Wtarget = floor(Ws * r / q) * q
Htarget = floor(Hs * r / q) * q
Wmodel = Ws * s
Hmodel = Hs * s
```

Never enlarge past the selected box. The model runs at its native scale; exact final dimensions are reached with one named Lanczos resize if needed. Padding/cropping and clean-aperture decisions are recorded.

### 5.3 Canonical MVP timeline

Raw RGB bytes carry no timestamp. Therefore the MVP freezes a rational progressive CFR timeline before scenes, diagnostics, or chunks:

```text
SourceFrameTiming(source_index, best_effort_pts, duration_pts, source_time_base,
                  repeat_pict, interlaced, top_field_first)
OutputFrameSpec(output_index, output_pts, output_duration, output_time_base,
                source_indexes, interpolation_fraction)
TimelinePlan(source_time_base, output_time_base, output_fps,
             timing_transform, source_frames, output_frames, sha256)
```

Rules:

- Supported CFR keeps its exact rational rate, including 30000/1001.
- Supported VFR is converted to a user-visible selected CFR; it is never silently called preserved.
- `bwdif` mode/parity/rate are explicit. No reliance on its default field-rate behavior.
- `fieldmatch,decimate` is planned as a continuous timing transform. If safe cadence boundaries cannot be proven, create progressive compressed mezzanine segments before SR rather than restart IVTC at arbitrary chunks.
- Stateful filters receive pre/post-roll; only the normalized core is retained.
- Raw decoder output is bound to `OutputFrameSpec` by exact index after explicit `fps,settb,setpts`. Byte count and decoded frame count must match.
- Encoder stdin declares `-f rawvideo -pixel_format rgb24 -video_size WxH -framerate NUM/DEN`; the output uses the Kaggle-compatible `-vsync 0` passthrough and frozen rational `-enc_time_base` contract, then validates time base, frame count, and PTS. Active-FFmpeg integration tests, not option-name assumptions, are authoritative.
- A true-VFR future path requires a timestamped transport such as NUT/PyAV and is out of MVP scope.

Scene and chunk coordinates are normalized output coordinates with a reverse source-PTS mapping. A chunk identity includes source decode range, normalized core range, timing plan hash, and context.

## 6. Source analysis

Analysis has two layers:

1. a full-duration low-cost proxy scan at deterministic cadence;
2. a 48-96 item stratified sample set selected from that scan.

The proxy scan records scene score, spatial/temporal information, luma/chroma statistics, edge density, noise proxy, blockiness/ringing, banding, motion, flat/dark/highlight coverage, text/line confidence, face confidence, and frozen/black/repeated-frame indicators.

Sampling is deterministic from the source hash and covers:

- opening, middle, closing;
- every major scene class and both sides of cuts;
- highest/lowest motion and spatial detail;
- flat gradients and dark areas;
- faces, text, thin lines, animation-like content;
- worst compression/ringing/noise;
- any interlace/telecine anomaly.

The report contains raw evidence and confidence, not only labels. Recipe routing may recommend candidates, but the user preview decides among eligible recipes.

## 7. Model and tool evaluation

| Model/tool | Role | Status | Mandatory gates |
|---|---|---|---|
| FFmpeg/FFprobe | Decode, normalize, encode, inspect, remux | Core | Recorded binary/license/build; exact media tests |
| Lanczos | Deterministic resize | Phase 0/core | Pixel and timing regression |
| Real-ESRGAN x2plus | 1080p to 4K spatial baseline | Phase 1 candidate | Safe weights/license, parity, VRAM/time/quality |
| Real-ESRGAN x4plus | General x4 challenger | Preview candidate | Same gates; exact final resize |
| Real-ESRGAN x4plus-anime | Animation candidate | Preview candidate | Content and quality gate |
| Real-ESRGAN general x4v3 | Compact/DNI candidate | Preview candidate | DNI parity and artifact gate |
| Spandrel | Safe model descriptor adapter | Core dependency candidate | Pinned API/environment; no runtime pickle |
| Video2X | Prior-art/orchestration reference | Reference only | No production dependency |
| RIFE | Frame-rate interpolation | Phase 2 optional | Python/Torch smoke, exact timeline, cut safety |
| RealBasicVSR | Real-world temporal SR | Phase 3 experimental | Custom bounded adapter, separate lock/license |
| NanoVSR, RealViFormer | Efficient temporal challengers | Phase 3 research | Reproducible code/weights/license and benchmark |
| DOVE, Stream-DiffVSR | Diffusion/streaming research | Watchlist | Hardware/memory/code maturity |
| LiteVSR | Large-model research | Watchlist | Noncommercial/weight and memory gates |
| InfVSR | Paper/repository tracking | Watchlist | Runnable public implementation |
| FlashVSR | Fast diffusion research | Watchlist | Block-sparse hardware path and exact Kaggle proof |

Real-ESRGAN remains a candidate until its weight terms are verified for the intended use. A repository code license does not automatically license a checkpoint.

### 7.1 Safe spatial adapter

Normal production runtime accepts only allowlisted `.safetensors`. Spandrel's public descriptor owns model-required size padding. EngVit separately owns:

- `context_pad`: pixels added around an inference crop for context;
- `blend_overlap`: neighboring output overlap used for seam blending.

The frame contract is deliberately single-image:

```python
class FrameEnhancer(Protocol):
    @property
    def scale(self) -> int: ...
    def enhance(self, frame_rgb: np.ndarray) -> np.ndarray: ...
    def close(self) -> None: ...
```

Input is contiguous `uint8 [H,W,3]` RGB. The adapter converts once to float `[1,3,H,W]` in `[0,1]`, with no invented mean/std normalization. Output returns RGB, clamps and rounds once, and preserves explicit color policy. Batch processing loops through the audited single-image contract unless the descriptor documents a safe batch contract.

DNI for compact general x4v3 is:

```text
merged = denoise_strength * general
       + (1 - denoise_strength) * weak_denoise
```

It runs in FP32 on CPU only after both wrappers are canonicalized and all keys, shapes, dtypes, scales, and non-floating tensors are reconciled. Endpoint and midpoint results must match the official Real-ESRGAN path within calibrated tolerance.

Legacy pickle conversion, if unavoidable, is a separate networkless/no-media/no-secret process using a patched PyTorch, `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`, CPU-only loading, resource limits, tensor-only validation, and safetensors output. The source file is never loaded in the notebook runtime.

### 7.2 Optional interpolation

`FrameInterpolator` is separate from enhancement:

```python
class FrameInterpolator(Protocol):
    def interpolate(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray,
        fractions: tuple[Fraction, ...]
    ) -> tuple[np.ndarray, ...]: ...
    def close(self) -> None: ...
```

RIFE runs after deinterlace/timing normalization and normally before spatial SR to save work. It never interpolates across a cut: the cut tick selects the first frame of the new scene. Pair buffers are cleared. Its exact upstream code archive/commit, license, weights, and environment are pinned offline. If current Kaggle Python/Torch/CUDA does not pass, the feature is unavailable rather than repairing the environment during a user job.

## 8. Tiling and memory

The tile engine:

1. calibrates representative bright, dark, high-detail, flat, text, and noise frames;
2. begins with a conservative candidate based on free VRAM;
3. applies descriptor-required size padding, EngVit context, and symmetric reflection where valid;
4. runs native model scale;
5. removes padding precisely;
6. blends overlaps with normalized separable cosine weights;
7. retries only the failed tile on CUDA OOM, clears cache, reduces tile size, and records the attempt;
8. fails if the minimum safe tile cannot run.

FP16 is enabled only per model/GPU after richer-corpus parity, NaN/Inf, clipping, seam, and visual tests. Queue bounds are bytes, not frame counts: a single 8K RGB24 frame is about 99.5 MB.

Tile QA compares tiled and untiled references where possible, multiple shifted grid origins, boundary versus interior errors, procedural patterns, and FP16/FP32. Numeric seam limits are calibrated, not guessed.

## 9. Chunks, coordinator, and resume

Chunks are scene-aligned when that does not violate timing-filter state. Each contains:

- source decode start/end PTS and context;
- normalized output start/end index/PTS and core/context;
- scene IDs and timing-plan hash;
- source full SHA-256;
- recipe/model/weight/precision/tile/final-resize hashes;
- encoder configuration/self-test hash;
- environment and application schema hashes.

Only the coordinator writes `job-manifest.json`. It assigns leases. Workers write a unique `.partial` segment and return immutable `ChunkCompletion`; the coordinator validates it, fsyncs, atomically renames, fsyncs the directory, and commits the next manifest generation. Startup recovers an orphan segment produced after rename but before manifest commit.

The atomic writer uses canonical UTF-8 JSON with sorted keys, compact separators, `allow_nan=false`, and explicit Fraction/Decimal/Path encodings. Volatile observations are excluded from identity hashes. Windows/OneDrive replacement has a bounded retry and validated backup path.

No decoder context, temporal window, interpolation pair, or latent state crosses a scene cut.

## 10. Encode, concat, and ancillary streams

### 10.1 Encoder admission

An `EncoderConfig` freezes codec, binary, hardware device, pixel format, rate control, bitrate/quality, preset, GOP, B-frames/references, time base, frame rate, color tags, and self-test hash. Hardware 10-bit HEVC uses an admitted P010 path; software x265 uses an admitted `yuv420p10le` path. An unsupported conversion or encoder pixel format is a refusal, not a silent downgrade.

Admission has:

- a fast small smoke test;
- a 2-4 frame exact target-geometry/pixel-format test;
- for 8K, at least two exact 7680x4320 high-entropy HEVC Main10 GOPs using final settings, followed by software decode and hardware decode when available.

P100 may expose different hardware encode capability than T4/L4, and some data-center GPUs have no NVENC. The pipeline probes, never infers from CUDA availability. If exact hardware admission fails, it benchmarks the exact software encoder; if time admission then fails, the job is rejected.

### 10.2 Segment contract and concat

Each encoder template guarantees an IDR first frame, closed GOP, fixed key interval, disabled adaptive scene cuts, identical B-frame/reference behavior, and repeated headers when needed. Options are confirmed with the active `ffmpeg -h encoder=...`.

Before stream-copy concat, validate:

- stream count/order, codec/tag, profile, level;
- dimensions, SAR, pixel format, field order;
- frame rate and time base;
- color range/matrix/transfer/primaries;
- encoder settings and codec extradata hash.

The concat list contains fixed, generated relative filenames, `-safe 1`, and exact rational duration directives. FFmpeg uses `-nostdin`. Any mismatch reprocesses or fails; it does not silently concatenate.

After concat, scan all packet DTS/PTS for monotonicity and boundary gaps/overlaps, decode all frames, verify exact count/duration, and compare the first/last two frame hashes of every boundary.

### 10.3 Presentation origin and stream policy

One rational presentation origin is applied to video, audio, subtitles, and chapters. QA checks first/last PTS and original relative start/end offsets for every selected stream. `-shortest` is prohibited.

- MKV is the preservation-first container and can keep all compatible selected audio/subtitle/attachment streams.
- MP4 is the compatibility-first default; incompatible streams require a visible conversion/omit choice.
- Copy is preferred when valid; every transcode, omission, or semantic loss is recorded.
- Explicit maps, metadata maps, chapter maps, languages, dispositions, and defaults are used.

Metadata is deny-all by default. Allow only technical color, orientation-cleared aspect, language, and disposition. Owner, GPS, device, title, comment, source filename, arbitrary chapters, attachments, thumbnails, data streams, and command lines are excluded unless the user explicitly opts into a reviewed stream type. Canary tests prevent leakage.

## 11. Color and HDR

MVP supports SDR only. It explicitly converts declared source matrix/primaries/transfer/range into a defined model RGB working space, then converts model RGB to the selected output colorimetry and writes matching tags. Missing metadata that prevents a safe decision requires user choice or rejection.

PQ, HLG, Dolby Vision, ambiguous HDR, and an HDR/SDR mismatch stop before processing. There is no dormant boolean that implies tone mapping exists. A future HDR-to-SDR phase must use a named 16-bit/float linear-light recipe, declared peak assumptions and tone-map algorithm, BT.709 output, 10-bit encode where required, and independent color QA.

## 12. Benchmark, admission, and storage

The benchmark runs the exact `ExecutionPlan` on samples spanning motion/detail/content classes and includes decode, timing filters, neural model, tiling, final resize, color conversion, encoder, and actual worker concurrency.

Let:

```text
N = planned output frames
F = measured end-to-end frames/second at the tested concurrency
Toverhead = measured startup + concat + remux + QA + persistence estimate
Tpred = N / F + Toverhead
Tneed = Tpred * uncertainty_factor + safety_margin
```

Admission requires known observed `remaining_session_seconds` and `Tneed <= remaining`. Unknown time rejects a full neural run. Time is rechecked before every chunk and finalization. Confidence/uncertainty grows when benchmark coverage is weak or performance variance is high.

Peak disk is computed by phase, not a single multiplier:

```text
analysis_peak = reports + proxies + samples
processing_peak = completed_segments + active_partial + manifests + logs
finalize_peak = segments + concatenated_video + remux_temp + final + QA
persistence_peak = finalize_peak + continuation_or_delivery_archive
required_free = max(all phase peaks) * safety_factor
```

Inputs and read-only weights are not copied if an attached path can be safely read. Intermediates are compressed; raw frames are never materialized as a full sequence.

### Cross-session continuation

`/kaggle/working` is ephemeral. A continuation bundle is complete only after:

1. its allowlisted files and hashes validate locally;
2. it is persisted to a private versioned Dataset/notebook output or explicitly downloaded;
3. the remote version/receipt is recorded;
4. the next session attaches the exact version and rehashes it.

Only then may the previous session remove continuation segments. The UI exposes “Prepare continuation,” “Resume continuation,” and “Run next slice.”

## 13. Quality system

### 13.1 Evidence states

Every required gate is `PASS`, `FAIL`, or `NOT_EVALUATED`. Missing libraries, models, source references, or thresholds produce `NOT_EVALUATED`, never `SKIP` or implicit success. Each metric result has an evidence ID, input hashes, implementation/model version, preprocessing, pooling, and calibration version.

### 13.2 Three non-interchangeable protocols

1. **Synthetic HR fidelity:** known clean HR versus restoration of a recorded second-order degradation. Measures PSNR, SSIM, LPIPS, temporal metrics, color, hallucination/artifact review.
2. **Encoder roundtrip:** pre-encode enhanced frames versus decoded final. Measures VMAF-NEG, CAMBI, frame/PTS integrity, color, and encode damage.
3. **Blind real-source consistency:** source versus downscaled output. Detects gross drift, flicker, color/range changes, and instability; it can never prove restoration or yield an overall pass.

LPIPS uses the pinned `alex` v0.1 contract with RGB `[-1,1]`. PSNR/SSIM specify RGB/Y domain, transfer, bit depth, window, border, and pooling. VMAF records model/hash, NEG mode, resolution, PTS alignment, and pooling. Temporal evidence includes TecoGAN-derived tOF/tLP on reference fixtures plus blind motion-compensated residual, flat-region flicker, freeze, and repeated-frame diagnostics. Color uses ROI and full-frame CIEDE2000 distributions. CAMBI covers source banding and encoder-added banding.

### 13.3 Sampling and human review

The deterministic sample set is used for fast iteration; full-duration proxy/structural scans catch localized failure. Human review uses synchronized/blinded A/B candidates, 100% crops, motion clips, stable labels, and a recorded rubric. A formal quality claim requires a P.910/BT.500-aligned screened study (at least 24 retained observers in the planned protocol); smaller review is labeled pilot/informal.

Numeric pass thresholds start `UNSET`. They are learned on a versioned calibration/development corpus, frozen, and then evaluated once on a locked holdout. Thresholds cannot be tuned after holdout inspection.

Synthetic/reference evidence qualifies a versioned model, recipe, tile policy, precision, environment, and source domain; it is attached to each user job rather than recomputed against a source that has no HR ground truth. Per-job QA requires structural/timing, final-encode roundtrip, color/banding/seam checks, full-duration blind diagnostics, and linkage to still-valid release evidence. A job fails if that evidence is absent, stale, or outside its calibrated source domain.

### 13.4 Structural release gates

- selected stream, codec/container policy, resolution, SAR, color, and rotation correct;
- CFR exact frame count and rational PTS plan; no non-monotonic PTS/DTS, gap, overlap, black insertion, freeze, or repeated-frame anomaly;
- signed A/V offset at start/middle/end, with pipeline-added skew no more than half an output frame;
- every selected ancillary stream accounted for;
- segment boundaries decode and hash correctly;
- no NaN/Inf, OOM retry corruption, tile seam failure, or metadata canary leak;
- all required quality evidence evaluated and passing calibrated thresholds.

## 14. Privacy, security, and licensing

Sensitive-media preflight fails closed unless it can verify or obtain a recorded user attestation that:

- notebook and media/weight/continuation Datasets are private/approved;
- internet is disabled;
- outputs will be persisted only to approved private destinations.

Visibility is rechecked before persistence. Logs use generated IDs, never source filenames or full paths. Cell outputs can be cleared. Delivery exports are allowlisted and exclude source media, segments, crops, proxies, logs, weights, paths, and diagnostics unless explicitly selected. Privacy is reduced exposure, not confidential computing: the cloud host still processes the media.

The model registry separately records code license and weight status (`verified`, `unverified`, `prohibited`), source URLs, hashes, terms URL/snapshot, reviewer/date, intended use, commercial/redistribution permissions, and notices. Unverified or incompatible weight terms block model selection.

FFmpeg licensing is evaluated from the actual binary's `-L` and `-buildconf`, intended invocation/distribution, and codec options. GPL/nonfree combinations receive a release review. DRM messaging says “encrypted, unsupported, or undecodable; no decryption/circumvention.”

## 15. Delivery phases

### Phase 0: deterministic media truth

Environment/privacy preflight, source selection, complete probe/timeline, geometry/color/orientation, diagnostics, canonical artifacts, coordinator/manifest, streaming Lanczos segments, concat/remux, structural QA, and private continuation.

**Gate:** exact short-fixture and long-plumbing matrix passes, including VFR-to-CFR, interlace/IVTC, offsets, rotations/SAR, stream preservation, kill/resume, and persistence.

### Phase 1: spatial neural release

Safe weight conversion/registry, Spandrel adapter, Real-ESRGAN parity, overlap tiles, previews, exact benchmark/admission, notebook/CLI, calibrated quality system, and real full-duration acceptance.

**Gate:** both real-model 15-minute 4K jobs pass on the minimum supported Kaggle GPU with complete evidence. The first baseline attempt is P100. If P100 fails or cannot be obtained for the recorded matrix, the public capability is narrowed to the exact passing GPU class; the plan does not relabel an untested faster GPU as a general Kaggle guarantee.

### Phase 2: optional timing and source conditions

RIFE environment package, exact interpolation timeline, cut safety, and source-condition enhancements whose transforms are fully specified.

**Gate:** feature-specific structural, temporal, visual, time, and license gates; failure leaves Phase 1 usable.

### Phase 3: temporal challengers

RealBasicVSR bounded-window adapter and separate lock; NanoVSR and RealViFormer evaluation.

**Gate:** a challenger must beat the spatial baseline on locked quality evidence while meeting the same session/disk/security contract.

### Phase 4: guarded 8K and research

Exact 8K HEVC Main10 encode/decode, model tiling, concurrent throughput, multi-session transaction, and current model watchlist.

**Gate:** `release-capabilities.json` enables 8K only for the exact passing environment/recipe. Otherwise the UI refuses it.

## 16. Definition of implementation-ready

This design is implementation-ready because it defines:

- the supported product and honest exclusions;
- source, path, stream, timing, geometry, color, and presentation contracts;
- safe dependency/checkpoint/license policy;
- model roles and promotion gates;
- bounded tiling, chunks, coordinator, concat, resume, and persistence semantics;
- benchmark equations and fail-closed admission;
- calibrated quality protocols rather than invented numbers;
- user flow, artifact tiers, privacy, and recovery;
- real full-duration release evidence.

It intentionally does **not** mark the product production-ready. That state is earned only by the companion implementation plan's recorded acceptance gates.

## 17. Primary references

### Platform and media

- [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [KaggleHub](https://github.com/Kaggle/kagglehub/blob/main/README.md)
- [Kaggle Docker Python](https://github.com/Kaggle/docker-python)
- [FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html)
- [FFmpeg command documentation](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)
- [NVIDIA FFmpeg transcoding guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)
- [NVIDIA NVENC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html)

### Models and loaders

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Spandrel](https://github.com/chaiNNer-org/spandrel)
- [Practical-RIFE](https://github.com/hzwer/ECCV2022-RIFE)
- [RealBasicVSR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.html)
- [MMagic RealBasicVSR documentation](https://mmagic.readthedocs.io/en/latest/model_zoo/video_super_resolution.html)
- [NanoVSR](https://github.com/filippawlicki/nanovsr)
- [RealViFormer](https://github.com/Yuehan717/RealViformer)
- [DOVE](https://github.com/zhangchen98/DOVE)
- [InfVSR](https://github.com/youngat1999/InfVSR)
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
- [Video2X](https://github.com/k4yt3x/video2x)

### Safety and quality

- [PyTorch serialization](https://pytorch.org/docs/stable/notes/serialization.html)
- [SafeTensors](https://github.com/huggingface/safetensors)
- [VMAF](https://github.com/Netflix/vmaf)
- [CAMBI](https://github.com/Netflix/vmaf/blob/master/resource/doc/cambi.md)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [TecoGAN](https://github.com/thunil/TecoGAN)
- [ITU-T P.910](https://www.itu.int/rec/T-REC-P.910)
- [ITU-R BT.500](https://www.itu.int/rec/R-REC-BT.500)
