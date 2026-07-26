from __future__ import annotations

from engvit.planning.disk import DiskInputs, estimate_peak_disk


def test_disk_estimate_uses_peak_phase_not_only_final_file() -> None:
    estimate = estimate_peak_disk(
        DiskInputs(
            source_bytes=1_000,
            segment_bytes=4_000,
            final_video_bytes=5_000,
            analysis_bytes=500,
            continuation_bytes=4_500,
            temporary_multiplier=1.2,
        )
    )
    assert estimate.required_bytes == max(
        estimate.analysis_peak_bytes,
        estimate.processing_peak_bytes,
        estimate.finalization_peak_bytes,
        estimate.persistence_peak_bytes,
    )
    assert estimate.required_bytes > estimate.final_video_bytes

