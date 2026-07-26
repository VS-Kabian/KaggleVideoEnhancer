"""Immutable public domain types shared by EngVit subsystems."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import gcd
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
MetricState: TypeAlias = Literal["PASS", "FAIL", "NOT_EVALUATED"]
HDRPolicy: TypeAlias = Literal["reject"]
FPSPolicy: TypeAlias = Literal["source_cfr", "normalize_cfr", "rife"]
ContainerPolicy: TypeAlias = Literal["mp4_compatibility", "mkv_preservation"]


@dataclass(frozen=True)
class Rational:
    """Normalized, exact rational used for frame rates and time bases."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator == 0:
            raise ValueError("Rational denominator must not be zero")
        numerator = self.numerator
        denominator = self.denominator
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        common = gcd(abs(numerator), denominator)
        object.__setattr__(self, "numerator", numerator // common)
        object.__setattr__(self, "denominator", denominator // common)

    def as_float(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True)
class JobPaths:
    root: Path
    artifacts: Path
    segments: Path
    partials: Path
    reports: Path
    previews: Path
    evidence: Path


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None
    time_base: Rational | None
    start_pts: int | None
    duration_pts: int | None
    disposition: dict[str, bool]
    language: str | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class VideoStreamInfo(StreamInfo):
    coded_width: int
    coded_height: int
    sample_aspect_ratio: Rational | None
    avg_frame_rate: Rational | None
    real_frame_rate: Rational | None
    pixel_format: str | None
    bits_per_raw_sample: int | None
    field_order: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    display_matrix: tuple[float, ...] | None
    side_data_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class MediaInfo:
    source: Path
    source_sha256: str
    format_name: str
    duration_seconds: Decimal | None
    streams: tuple[StreamInfo, ...]
    selected_video_index: int


@dataclass(frozen=True)
class SourceFrameTiming:
    source_index: int
    best_effort_pts: int
    duration_pts: int
    source_time_base: Rational
    repeat_pict: int
    interlaced: bool
    top_field_first: bool | None


@dataclass(frozen=True)
class OutputFrameSpec:
    output_index: int
    output_pts: int
    output_duration: int
    source_indexes: tuple[int, ...]
    interpolation_fraction: Rational | None


@dataclass(frozen=True)
class TimelinePlan:
    source_time_base: Rational
    output_time_base: Rational
    output_fps: Rational
    timing_transform: tuple[str, ...]
    source_frames: tuple[SourceFrameTiming, ...]
    output_frames: tuple[OutputFrameSpec, ...]
    sha256: str


@dataclass(frozen=True)
class GeometryPlan:
    coded_size: tuple[int, int]
    oriented_size: tuple[int, int]
    target_size: tuple[int, int]
    model_size: tuple[int, int]
    input_sar: Rational | None
    output_sar: Rational
    pixel_transform: tuple[str, ...]
    final_resize: Literal["lanczos"] | None


@dataclass(frozen=True)
class DiagnosticReport:
    source_sha256: str
    timeline_sha256: str
    scan_rows_sha256: str
    sample_indexes: tuple[int, ...]
    features: dict[str, JSONValue]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class EncoderConfig:
    ffmpeg_sha256: str
    encoder: str
    hardware_device: int | None
    codec: str
    pixel_format: str
    rate_control: dict[str, JSONValue]
    preset: str
    gop: int
    b_frames: int
    output_fps: Rational
    output_time_base: Rational
    color: dict[str, str]
    self_test_sha256: str


@dataclass(frozen=True)
class TilePolicy:
    tile_size: int
    context_pad: int
    blend_overlap: int
    precision: Literal["fp32", "fp16"]
    device_id: int
    calibration_sha256: str


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    model_id: str
    model_scale: int
    denoise_strength: Decimal | None
    fps_policy: FPSPolicy
    final_resize: Literal["lanczos"] | None


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    source_decode_start_pts: int
    source_decode_end_pts: int
    output_core_start: int
    output_core_end: int
    context_before: int
    context_after: int
    scene_ids: tuple[int, ...]
    identity_sha256: str


@dataclass(frozen=True)
class ChunkCompletion:
    chunk_id: str
    lease_id: str
    identity_sha256: str
    partial_path: Path
    bytes: int
    sha256: str
    frame_count: int
    first_pts: int
    last_pts: int
    boundary_frame_hashes: tuple[str, ...]
    encoder_extradata_sha256: str
    observations: dict[str, JSONValue]


@dataclass(frozen=True)
class BenchmarkResult:
    frames: int
    elapsed_seconds: Decimal
    end_to_end_fps: Decimal
    peak_vram_bytes: int
    peak_disk_bytes: int
    worker_count: int
    variance: Decimal


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: str
    source_sha256: str
    selection_sha256: str
    timeline_sha256: str
    diagnostic_sha256: str
    recipe: Recipe
    geometry: GeometryPlan
    tiles: tuple[TilePolicy, ...]
    encoder: EncoderConfig
    chunks: tuple[ChunkSpec, ...]
    benchmark: BenchmarkResult
    environment_sha256: str
    required_disk_bytes: int
    predicted_seconds: int
    safety_seconds: int
    identity_sha256: str


@dataclass(frozen=True)
class MetricEvidence:
    evidence_id: str
    protocol: Literal[
        "synthetic_hr_fidelity",
        "encoder_roundtrip",
        "blind_real_source_consistency",
        "structural",
    ]
    state: MetricState
    metric: str
    value: JSONValue
    threshold_version: str | None
    inputs: dict[str, str]
    implementation: dict[str, str]
    reason: str | None


class FrameEnhancer(Protocol):
    @property
    def scale(self) -> int: ...

    def enhance(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]: ...

    def close(self) -> None: ...


class FrameInterpolator(Protocol):
    def interpolate(
        self,
        left_rgb: NDArray[np.uint8],
        right_rgb: NDArray[np.uint8],
        fractions: tuple[Rational, ...],
    ) -> tuple[NDArray[np.uint8], ...]: ...

    def close(self) -> None: ...
