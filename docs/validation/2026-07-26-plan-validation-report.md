# EngVit Plan Validation Report

**Validated:** 2026-07-26  
**Scope:** Kaggle-hosted enhancement and upscaling of user-supplied 10-15 minute videos  
**Artifacts reviewed:** technical design and implementation plan dated 2026-07-26  
**Method:** eight independent specialist reviews, reconciliation against primary documentation and current upstream repositories, then a dependency and acceptance-gate audit

## Verdict

The project is technically implementable, with two important qualifications:

1. **4K is the release target, not a current guarantee.** It becomes supported only after the exact production pipeline completes the two full-duration acceptance jobs in the implementation plan on the minimum supported Kaggle GPU. A synthetic or lightweight substitute does not satisfy that gate.
2. **8K is an experimental, per-job capability.** It stays disabled in the public release unless the active session passes exact-resolution model, memory, time, disk, encoder, decoder, and persistence checks. Free Kaggle availability is not stable enough to promise 8K generally.

The initial plan had a sound spatial-super-resolution direction but was not execution-ready. Its principal defects were timestamp semantics, ordering of timing transforms and chunking, unsafe checkpoint assumptions, unsupported quality thresholds, incomplete cross-session persistence, and acceptance tests that did not run a real model for a full 10-15 minute source. The revised documents close those design gaps at plan level. They do not claim the unbuilt system has passed runtime gates.

## Review method

The plan was challenged from eight independent perspectives:

| Review lane | Question tested |
|---|---|
| Kaggle feasibility | Can the promised workload finish within volatile GPU, session, disk, and encoder limits? |
| Spatial model loading | Are Real-ESRGAN-family checkpoints loaded safely and with pixel-equivalent preprocessing? |
| Temporal models | Are RIFE and newer VSR models current, correctly classified, and realistically runnable? |
| Media pipeline | Are timing, chunking, concat, audio, subtitle, rotation, SAR, and color contracts exact? |
| Quality validation | Can the system distinguish improvement from hallucination, flicker, seams, and encode damage? |
| Security and licensing | Can private media, dependencies, model weights, metadata, and pickle risks be controlled? |
| Implementation dependency audit | Are all interfaces defined and tasks ordered so an engineer can implement them without guessing? |
| Notebook UX and operations | Can a non-expert select, preview, run, pause, resume, persist, and retrieve a result safely? |

Primary sources were preferred: Kaggle documentation and images, FFmpeg documentation, NVIDIA's codec documentation, upstream repositories, official papers, model cards, PyTorch security guidance, and ITU recommendations.

## Critical findings and dispositions

### 1. Raw RGB bytes do not carry timestamps

**Finding:** An elementary `rgb24` pipe contains pixels only. The original `DecodedFrame.pts` promise was false, and a VFR source could not be preserved by attaching an inferred PTS to bytes after the fact.

**Disposition:** The MVP now has an explicit **CFR-normalization contract**:

- a complete streamed `ffprobe -show_frames` timing pass records source timing;
- the timing transform is frozen before scenes or chunks;
- FFmpeg applies explicit `fps`, `settb`, and `setpts`;
- each decoded raw frame is bound by validated output index to a persisted rational `TimelinePlan`;
- encoder rawvideo input always declares exact frame rate, size, and pixel format;
- VFR input is supported only by explicit conversion to the selected CFR output;
- true VFR output is deferred until a timestamped transport such as NUT/PyAV is implemented.

Interlace and telecine transforms are also frozen before chunk planning. Stateful filters receive context and are trimmed to a normalized core; arbitrary independent IVTC chunks are prohibited.

### 2. Chunk coordinates and concat were underspecified

**Finding:** Source-frame chunks could not remain correct after deinterlacing or decimation. Closed GOP alone was insufficient to guarantee stream-copy concatenation.

**Disposition:** Every chunk now carries both source-decode and normalized-output ranges. Segment identity includes the timeline hash, recipe, model, tile policy, precision, encoder self-test, and environment. Encoder-specific templates must enforce IDR-at-start, closed GOP, fixed GOP, disabled scene-cut insertion, and stable headers. Before concat, the coordinator compares codec parameters and codec-extradata hashes. It writes exact rational durations, scans packet PTS/DTS and gaps, decodes the result, and checks boundary-frame hashes. A mismatch fails closed.

### 3. Audio, subtitles, metadata, rotation, and color lacked a common presentation contract

**Finding:** End-duration checks could miss constant A/V offsets. FFmpeg autorotation plus manual rotation could rotate twice. Copying source color tags after an RGB transform could mislabel pixels. `rgb24` cannot implement controlled HDR.

**Disposition:**

- one rational presentation origin applies to video, audio, subtitles, and chapters;
- start and end offsets are checked per stream;
- the source's complete display matrix is applied once under `-noautorotate`, output is square-pixel `setsar=1`, and stale display metadata is cleared;
- SDR conversion uses an explicit declared working RGB space and matching output tags;
- PQ/HLG and ambiguous HDR are rejected in MVP; there is no misleading `allow_hdr_to_sdr` switch;
- metadata is deny-by-default, with only technical fields allowlisted;
- ancillary stream preservation is container-aware and losses are reported rather than hidden.

### 4. The production acceptance test was not a production test

**Finding:** The original full-duration test used a fake or lightweight enhancer, while real neural tests were short. It could prove plumbing, not the promised workload.

**Disposition:** Release now requires, on the minimum supported Kaggle GPU:

- an initial P100 baseline attempt; if it fails or cannot be included in the recorded matrix, capability is narrowed to the exact faster GPU class that passes rather than advertised as generic Kaggle support;
- a real 15-minute 1080p30 to 4K job using the Real-ESRGAN x2 baseline;
- a real 15-minute 720p30 to 4K job using the compact x4 baseline;
- full decode, neural enhancement, encode, remux, structural QA, visual evidence, interruption, persistence, and resume;
- representative compression, text, faces, motion, dark/flat regions, and scene cuts;
- measured end-to-end throughput and peak disk/VRAM.

For context, a 15-minute 30 fps source contains 27,000 frames. Before overhead, a 10.5-hour budget requires at least about 0.71 end-to-end fps; the revised admission controller computes the actual threshold from source frame count and observed remaining time rather than relying on this illustration.

### 5. Session time and persistence assumptions were unsafe

**Finding:** A fixed 10.5-hour session default could imply knowledge Kaggle does not expose, and `/kaggle/working` is not durable across sessions.

**Disposition:** `remaining_session_seconds` is optional and observed, never fabricated. If it is unknown, a neural full run is not admitted. It is rechecked before each chunk and finalization. Continuation is a transaction: produce a hash-verified continuation bundle, persist it to a private versioned Dataset or explicitly download it, attach the exact version in the next session, verify every hash, then resume. Cleanup follows verified persistence, never precedes it.

### 6. Dependency and checkpoint handling was not safe enough

**Finding:** Internet installs can silently replace Kaggle's Torch/CUDA stack. PyTorch pickle checkpoint formats can execute code. Weight licensing is distinct from repository code licensing.

**Disposition:**

- normal runtime accepts only audited `.safetensors`;
- legacy conversion occurs in an isolated, networkless, no-media, no-secret process with `weights_only`, resource limits, tensor inventory, source/output hashes, and parity tests;
- runtime rejects `.pth`, `.pt`, `.ckpt`, `.pkl`, VCS dependencies, and source distributions;
- the offline wheelhouse is binary-only, hash-locked, SBOM-recorded, and installed with `--no-index --require-hashes`;
- Torch, torchvision, CUDA, and NumPy versions/hashes are compared before and after installation;
- model code license and weight-use terms are separate gates; unverified weight terms block use for the intended release.

The revised Real-ESRGAN adapter uses Spandrel's public image descriptor contract, loads one `[1,C,H,W]` RGB `[0,1]` image at a time, distinguishes descriptor-required size padding from EngVit context padding and tile overlap, and performs DNI in FP32 on CPU only after key/shape/dtype validation.

### 7. Temporal/newer models were overpromised

**Finding:** RIFE is interpolation, not super-resolution, and its current upstream environment may not support the active Kaggle Python. RealBasicVSR's upstream path is legacy and whole-clip oriented. Several newer diffusion or streaming papers remain unsuitable for a baseline free-Kaggle promise.

**Disposition:** Temporal features cannot block the spatial MVP:

- **RIFE:** optional frame-rate transform only; gated by an offline environment smoke test, exact timeline mapping, and cut safety.
- **RealBasicVSR:** experimental challenger requiring a custom bounded-window adapter and separate dependency lock.
- **NanoVSR and RealViFormer:** Phase 3 benchmark challengers.
- **DOVE, Stream-DiffVSR, LiteVSR, InfVSR, FlashVSR:** research watchlist unless their exact code, weight, license, memory, and hardware gates pass. FlashVSR's current block-sparse attention requirements do not support a generic Kaggle claim.

No temporal context, interpolation pair, or latent state may cross a detected cut.

### 8. Quality thresholds were arbitrary

**Finding:** Fixed SSIM, LPIPS, color, temporal, and seam numbers had no calibrated corpus or locked protocol. Blind source consistency cannot prove perceptual restoration.

**Disposition:** Uncalibrated thresholds are `UNSET`, not guessed. Three separate protocols prevent invalid comparisons:

1. `synthetic_hr_fidelity`: known HR target against restored degraded input;
2. `encoder_roundtrip`: enhanced pre-encode frames against decoded final output;
3. `blind_real_source_consistency`: source against downscaled output, diagnostic only and never a pass signal.

Release evidence requires structural/timing checks, synthetic PSNR/SSIM/LPIPS, VMAF-NEG and CAMBI for the encode path, reference temporal metrics, blind temporal diagnostics, CIEDE2000 distributions, tile-boundary tests, and blinded human evidence. Missing required evidence yields `NOT_EVALUATED`, never `PASS`. Numeric thresholds are calibrated on a development corpus and frozen before a locked holdout is opened.

### 9. Multi-GPU manifest writes could lose results

**Finding:** Atomic file replacement prevents torn writes but not two workers overwriting each other's manifest updates.

**Disposition:** A single coordinator owns leases and the manifest. Workers write unique partial segments and return immutable `ChunkCompletion` records. The coordinator validates, fsyncs, renames, then advances a manifest generation. Recovery covers a crash after rename but before commit. Concurrent two-GPU benchmarking is mandatory before assuming scaling.

### 10. Notebook delivery and accessibility were incomplete

**Finding:** Users lacked a clear path from upload to durable result, and browser playback of high-resolution intermediates was assumed.

**Disposition:** The notebook flow is now:

`Preflight -> Select -> Analyze -> Configure -> Preview -> Benchmark -> Run/Resume -> QA/Persist`

It includes source filtering and provenance, private-visibility checks, compatible H.264/AAC preview proxies plus PNG crops, keyboard and screen-reader labels, ETA confidence, disk/VRAM status, stop-after-segment behavior, continuation controls, failure cards, explicit SHA-256 output receipts, and local/private-Dataset retrieval instructions.

## Model decision matrix

| Component | Revised status | Reason |
|---|---|---|
| Lanczos | Required Phase 0 | Deterministic plumbing and fallback baseline |
| Real-ESRGAN x2plus | Primary 1080p to 4K spatial baseline, pending gates | Mature blind SR; x2 avoids unnecessary 4x work |
| Real-ESRGAN x4plus | Preview challenger | Strong but expensive; resize to exact target |
| Real-ESRGAN x4plus-anime | Animation candidate | Content-specific; must win preview |
| Real-ESRGAN general x4v3 | Fast compact candidate | Useful for constrained runs and DNI |
| RIFE | Optional FPS transform | Not SR; environment and cut gates required |
| RealBasicVSR | Experimental temporal challenger | Legacy/custom bounded adapter required |
| NanoVSR / RealViFormer | Phase 3 challengers | Promising efficiency, not baseline |
| DOVE / Stream-DiffVSR / LiteVSR / InfVSR / FlashVSR | Research watchlist | Hardware, code maturity, memory, or license uncertainty |
| Video2X | Reference/orchestration comparison | Useful precedent, not the core runtime dependency |

No model is releasable merely because code loads. It must pass weight-license, safe-format, pixel-contract, VRAM, time, disk, quality, and exact-session acceptance gates.

## Residual risks that implementation must close

The plan is now ready to execute, but these are deliberately unresolved runtime facts:

- current Kaggle quota, GPU type, session limit, image, disk, and encoder capabilities;
- actual full-duration neural throughput;
- Real-ESRGAN weight-use terms for the intended distribution/use case;
- calibrated quality thresholds and locked holdout results;
- RIFE compatibility with the active Kaggle Python/Torch/CUDA image;
- reliable multi-session Dataset persistence under the user's account;
- 8K HEVC Main10 encode/decode at final settings.

Each appears as a fail-closed acceptance gate in the implementation plan. None is converted into an optimistic assumption.

## Sources used for decisive corrections

- [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle kernels CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [KaggleHub resources and outputs](https://github.com/Kaggle/kagglehub/blob/main/README.md)
- [Kaggle Docker Python image](https://github.com/Kaggle/docker-python)
- [FFmpeg formats: rawvideo and concat](https://ffmpeg.org/ffmpeg-formats.html)
- [FFmpeg frame-rate, mapping, autorotation, and metadata options](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg filters: bwdif, fieldmatch, decimate, colorspace, tonemap](https://ffmpeg.org/ffmpeg-filters.html)
- [NVIDIA Video Codec SDK FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)
- [NVIDIA NVENC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html)
- [Real-ESRGAN upstream repository](https://github.com/xinntao/Real-ESRGAN)
- [Spandrel upstream repository and API](https://github.com/chaiNNer-org/spandrel)
- [Practical-RIFE upstream repository](https://github.com/hzwer/ECCV2022-RIFE)
- [RealBasicVSR paper and code entry](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.html)
- [PyTorch serialization semantics](https://pytorch.org/docs/stable/notes/serialization.html)
- [SafeTensors format](https://github.com/huggingface/safetensors)
- [Netflix VMAF](https://github.com/Netflix/vmaf)
- [Netflix CAMBI](https://github.com/Netflix/vmaf/blob/master/resource/doc/cambi.md)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [TecoGAN temporal evaluation](https://github.com/thunil/TecoGAN)
- [ITU-T P.910](https://www.itu.int/rec/T-REC-P.910)
- [ITU-R BT.500](https://www.itu.int/rec/R-REC-BT.500)

## Validation conclusion

The revised design and implementation plan are coherent, dependency-ordered, fail-closed, and implementable without an engineer inventing missing behavior. “Ready to implement” means the work items and gates are specified. It does not mean 4K or 8K has already been demonstrated on Kaggle; those claims are intentionally withheld until the recorded release matrix passes.
