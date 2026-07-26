"""Recorded end-to-end benchmark summaries."""

from __future__ import annotations

from decimal import Decimal

from engvit.types import BenchmarkResult


def record_benchmark(
    *,
    frames: int,
    elapsed_samples: tuple[Decimal, ...],
    peak_vram_bytes: int,
    peak_disk_bytes: int,
    worker_count: int,
) -> BenchmarkResult:
    """Summarize repeated measured samples without hiding run variance."""
    if frames <= 0 or not elapsed_samples or any(value <= 0 for value in elapsed_samples):
        raise ValueError("benchmark requires positive frames and elapsed samples")
    if peak_vram_bytes < 0 or peak_disk_bytes < 0 or worker_count < 1:
        raise ValueError("benchmark resource observations are invalid")
    sample_count = Decimal(len(elapsed_samples))
    elapsed = sum(elapsed_samples, start=Decimal(0)) / sample_count
    variance = (
        sum(
            ((value - elapsed) ** 2 for value in elapsed_samples),
            start=Decimal(0),
        )
        / sample_count
    )
    deviation = variance.sqrt()
    return BenchmarkResult(
        frames=frames,
        elapsed_seconds=elapsed,
        end_to_end_fps=Decimal(frames) / elapsed,
        peak_vram_bytes=peak_vram_bytes,
        peak_disk_bytes=peak_disk_bytes,
        worker_count=worker_count,
        variance=deviation / elapsed,
    )
