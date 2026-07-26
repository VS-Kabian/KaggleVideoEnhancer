from __future__ import annotations

from pathlib import Path

import pytest

from engvit.paths import create_job_paths


def test_create_job_paths_builds_the_canonical_layout(tmp_path: Path) -> None:
    paths = create_job_paths(tmp_path, "job-20260726")
    assert paths.root == (tmp_path / "job-20260726").resolve()
    assert paths.artifacts.is_dir()
    assert paths.segments.is_dir()
    assert paths.partials.is_dir()
    assert paths.reports.is_dir()
    assert paths.previews.is_dir()
    assert paths.evidence.is_dir()


@pytest.mark.parametrize("job_id", ["../escape", "a/b", "a\\b", "", "."])
def test_create_job_paths_rejects_traversal_and_ambiguous_ids(
    tmp_path: Path, job_id: str
) -> None:
    """Catches a job ID escaping or aliasing the configured output root."""
    with pytest.raises(ValueError, match="job_id"):
        create_job_paths(tmp_path, job_id)

