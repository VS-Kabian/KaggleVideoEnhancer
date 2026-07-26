from __future__ import annotations

from engvit.progress import ProgressSnapshot


def test_progress_snapshot_requires_bounded_counts_and_eta_order() -> None:
    snapshot = ProgressSnapshot(
        phase="render",
        frames_complete=10,
        frames_total=100,
        chunks_complete=1,
        chunks_total=10,
        elapsed_seconds=20,
        eta_low_seconds=100,
        eta_high_seconds=140,
        confidence="medium",
        disk_used_bytes=1000,
        disk_free_bytes=2000,
        vram_used_bytes=500,
        vram_free_bytes=1500,
        last_checkpoint="chunk-000000",
        retry_count=0,
        expected_session_finish=True,
    )
    assert snapshot.percent == 10.0
