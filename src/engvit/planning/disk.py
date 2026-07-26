"""Peak-phase disk estimation."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class DiskInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_bytes: int = Field(ge=0)
    segment_bytes: int = Field(ge=0)
    final_video_bytes: int = Field(ge=0)
    analysis_bytes: int = Field(ge=0)
    continuation_bytes: int = Field(ge=0)
    temporary_multiplier: float = Field(ge=1.0, le=5.0)


class DiskEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_bytes: int
    final_video_bytes: int
    analysis_peak_bytes: int
    processing_peak_bytes: int
    finalization_peak_bytes: int
    persistence_peak_bytes: int
    required_bytes: int


def estimate_peak_disk(inputs: DiskInputs) -> DiskEstimate:
    """Estimate concurrent bytes by lifecycle phase, including temporary copies."""
    analysis = inputs.source_bytes + inputs.analysis_bytes
    processing = (
        inputs.source_bytes
        + inputs.analysis_bytes
        + math.ceil(inputs.segment_bytes * inputs.temporary_multiplier)
    )
    finalization = (
        inputs.source_bytes
        + inputs.segment_bytes
        + math.ceil(inputs.final_video_bytes * inputs.temporary_multiplier)
    )
    persistence = (
        inputs.segment_bytes
        + inputs.final_video_bytes
        + math.ceil(inputs.continuation_bytes * inputs.temporary_multiplier)
    )
    required = max(analysis, processing, finalization, persistence)
    return DiskEstimate(
        source_bytes=inputs.source_bytes,
        final_video_bytes=inputs.final_video_bytes,
        analysis_peak_bytes=analysis,
        processing_peak_bytes=processing,
        finalization_peak_bytes=finalization,
        persistence_peak_bytes=persistence,
        required_bytes=required,
    )

