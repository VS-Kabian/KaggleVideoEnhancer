"""Creation of contained, canonical per-job directories."""

from __future__ import annotations

import re
from pathlib import Path

from engvit.types import JobPaths

_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def create_job_paths(root: Path, job_id: str) -> JobPaths:
    """Create the fixed artifact layout without allowing path traversal."""
    if not _JOB_ID.fullmatch(job_id) or job_id in {".", ".."}:
        raise ValueError("job_id must be a safe 1-128 character identifier")

    approved_root = root.resolve(strict=False)
    job_root = (approved_root / job_id).resolve(strict=False)
    if job_root.parent != approved_root:
        raise ValueError("job_id resolves outside the approved root")

    directories = {
        "artifacts": job_root / "artifacts",
        "segments": job_root / "segments",
        "partials": job_root / "partials",
        "reports": job_root / "reports",
        "previews": job_root / "previews",
        "evidence": job_root / "evidence",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return JobPaths(root=job_root, **directories)
