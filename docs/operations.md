# Operations

## State model

Jobs move through `running -> pause_requested -> paused -> running -> complete`.
Only the coordinator may advance the atomic manifest generation. A pause occurs
after a segment commit; it never abandons a half-written segment as complete.

On startup, recovery:

1. verifies the manifest belongs to the same source/config/environment identity;
2. expires stale leases;
3. revalidates completed segment paths, sizes, hashes, frame counts, and PTS;
4. accepts a staged completion sidecar only if all invariants hold;
5. resets invalid completions to pending.

## CLI workflow

All commands emit compact JSON and redact source paths in failures.

```text
engvit preflight --context configs/kaggle-private-context.example.json
engvit discover --dataset-root /kaggle/input/private-media --handle owner/media --version 1
engvit analyze --request configs/kaggle-request.example.json
engvit preview --request configs/kaggle-request.example.json
engvit benchmark --output-root /kaggle/working/bench
engvit run --request configs/kaggle-request.example.json --max-new-chunks 1
engvit resume --request configs/kaggle-request.example.json
engvit qa --request configs/kaggle-request.example.json
engvit persist --output-root /kaggle/working/engvit-jobs --job-id engvit-job --archive /kaggle/working/engvit-job.zip
```

`pause` requests a manifest-safe stop. It does not kill an active FFmpeg
process. The notebook’s `MAX_NEW_CHUNKS` option is the preferred single-worker
“stop after segment” control.

## Failure handling

- Do not edit `manifest.json`, completion sidecars, or segment files.
- Keep the original Dataset handle, version, and source bytes unchanged.
- Do not resume after replacing Python, FFmpeg, Torch, NumPy, model weights, or
  configuration; start a new job ID.
- Preserve the job directory before Kaggle expiry.
- Treat missing session time, disk, VRAM, encoder admission, model approval, or
  quality thresholds as refusal—not as a warning.
