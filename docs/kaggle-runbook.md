# Kaggle runbook

## 1. Package inputs privately

Create two private Kaggle Datasets:

1. **Code Dataset**: this repository, preserving `src/`, `notebooks/`,
   `release-capabilities.json`, `models/`, `quality/`, and `dependencies/`.
2. **Media Dataset**: the uploaded source video. Do not place credentials,
   remote URLs, cookies, or unrelated private files beside it.

Keep the Notebook private and turn Internet **Off** before attaching sensitive
media. EngVit does not download weights at runtime.

## 2. Prove the environment with the smoke job

Open `notebooks/engvit_kaggle.ipynb` from the code Dataset and run every cell
unchanged. Expected final facts:

- state `complete`;
- 2 completed chunks;
- 6 frames with PTS 0 through 5;
- time base `1/10`;
- every structural metric `PASS`.

This proves only the notebook/FFmpeg plumbing on that runtime.

## 3. Configure a real Phase 0 job

In Section 2:

- set `SMOKE_MODE = False`; this automatically sets `RUN_JOB = False`;
- set the attached Dataset handle, version, root, and relative video path;
- set both visibility fields to the facts observed in Kaggle;
- if visibility cannot be queried, add a non-empty
  `VISIBILITY_ATTESTATION` only after manually verifying the Notebook and every
  attached Dataset are private;
- keep `INTERNET_ENABLED = False`;
- choose a target no larger than 3840x2160;
- use `source_cfr`, or set both target-FPS numbers with `normalize_cfr`.

Run Sections 1–5. Review the source hash prefix, duration, selected stream,
frame count, output rate, color decision, target geometry, and chunk count.
Preparation refuses unsafe paths, media over 15 minutes, unsupported/missing SDR
color metadata, HDR/Dolby Vision/BT.2020, unsafe stream selection, unsupported
telecine chunking, invalid timing, and targets above 4K.

## 4. Run in bounded slices

Set:

```python
RUN_JOB = True
MAX_NEW_CHUNKS = 1
RESUME_PAUSED = False
```

Run Section 6. The worker commits one fully encoded and decoded segment and then
pauses. Review `manifest.json`, free disk, elapsed time, and remaining Kaggle
session time. Increase `MAX_NEW_CHUNKS` only after measured evidence indicates
the slice fits.

To continue a paused job in the same attached state, set
`RESUME_PAUSED = True` and run Section 6 again. A chunk is reused only when its
identity, path containment, bytes, SHA-256, frame count, PTS range, and encoder
identity verify.

## 5. Preserve work across sessions

Set `PREPARE_CONTINUATION = True` after at least one committed segment.
Section 7 creates a deterministic ZIP and prints its byte count and SHA-256.
Upload that ZIP only to a private Dataset or retrieve it to a private local
destination. `/kaggle/working` is temporary.

On a new session, verify the archive receipt and restore it with
`engvit.persistence.resume_continuation`; do not copy individual segment files
by hand. The current notebook exposes packaging but cross-session restoration is
an operator/API step pending a dedicated Kaggle Dataset publisher.

## 6. Retrieve and interpret the result

The current delivery artifact is `enhanced-video.mkv` containing H.264 video.
The structural gate validates exact frame count, contiguous logical PTS,
dimensions, time base, square-pixel SAR, and boundary hashes. Audio/subtitle/
attachment preservation and MP4 delivery are not enabled in this Phase 0
notebook.

A `PASS` here is a structurally correct deterministic resize—not proof of
neural detail recovery, perceptual superiority, or production-qualified 4K.
