from __future__ import annotations

from decimal import Decimal

from engvit.planning.benchmark import record_benchmark


def test_benchmark_records_end_to_end_rate_and_variance() -> None:
    result = record_benchmark(
        frames=300,
        elapsed_samples=(Decimal("30"), Decimal("33"), Decimal("27")),
        peak_vram_bytes=2_000,
        peak_disk_bytes=3_000,
        worker_count=1,
    )
    assert result.end_to_end_fps == Decimal("10")
    assert result.variance > 0

