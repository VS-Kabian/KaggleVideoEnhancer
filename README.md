# EngVit

EngVit is a fail-closed, resumable video-upscaling pipeline designed for private
Kaggle inputs. The current executable release delivers a deterministic Lanczos
baseline with exact frame-timing analysis, chunk verification, pause/resume,
structural QA, and continuation archives.

The implementation does **not** yet claim production-qualified neural 4K or 8K:

- Real-ESRGAN-family registry entries are disabled until their exact code,
  license, SafeTensors hash, loader parity, and Kaggle acceptance evidence pass.
- RIFE and temporal-VSR adapters are disabled and cannot import model code.
- 8K cannot be submitted through the current Kaggle request model.
- `release-capabilities.json` is all false until the full acceptance matrix has
  independently audited receipts.

## Fastest safe start

1. Open [notebooks/engvit_kaggle.ipynb](notebooks/engvit_kaggle.ipynb).
2. Run it unchanged. Its default six-frame smoke job performs real FFmpeg
   decode, two verified chunks, concat, and structural QA.
3. Follow [docs/kaggle-runbook.md](docs/kaggle-runbook.md) before using private
   media.

For local development:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m mypy --strict src
```

The installed operator commands are:

```text
engvit preflight|discover|analyze|preview|benchmark|run|pause|resume|qa|persist
```

The `benchmark` command is intentionally labeled a six-frame plumbing smoke and
cannot admit a full job.

## Design and evidence

- [Validated technical design](docs/plans/2026-07-26-kaggle-video-upscaling-research-and-design.md)
- [Adversarial validation report](docs/validation/2026-07-26-plan-validation-report.md)
- [Dependency-ordered implementation plan](docs/superpowers/plans/2026-07-26-kaggle-video-upscaling-implementation.md)
- [Operations](docs/operations.md)
- [Privacy](docs/privacy.md)
- [License policy](docs/licenses.md)
- [Quality method](docs/quality-method.md)
- [Known limitations](docs/limitations.md)
