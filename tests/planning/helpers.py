from __future__ import annotations

from decimal import Decimal

from engvit.types import (
    BenchmarkResult,
    ChunkSpec,
    EncoderConfig,
    GeometryPlan,
    Rational,
    Recipe,
    TilePolicy,
)


def recipe(model_id: str = "lanczos") -> Recipe:
    return Recipe(
        recipe_id=f"{model_id}-recipe",
        model_id=model_id,
        model_scale=2,
        denoise_strength=None,
        fps_policy="source_cfr",
        final_resize=None,
    )


def geometry() -> GeometryPlan:
    return GeometryPlan(
        coded_size=(1920, 1080),
        oriented_size=(1920, 1080),
        target_size=(3840, 2160),
        model_size=(3840, 2160),
        input_sar=Rational(1, 1),
        output_sar=Rational(1, 1),
        pixel_transform=("setsar:1",),
        final_resize=None,
    )


def tile() -> TilePolicy:
    return TilePolicy(
        tile_size=256,
        context_pad=16,
        blend_overlap=32,
        precision="fp32",
        device_id=0,
        calibration_sha256="a" * 64,
    )


def encoder() -> EncoderConfig:
    return EncoderConfig(
        ffmpeg_sha256="b" * 64,
        encoder="libx264",
        hardware_device=None,
        codec="h264",
        pixel_format="yuv420p",
        rate_control={"mode": "crf", "crf": 18},
        preset="medium",
        gop=60,
        b_frames=0,
        output_fps=Rational(30, 1),
        output_time_base=Rational(1, 30),
        color={
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        },
        self_test_sha256="c" * 64,
    )


def chunks() -> tuple[ChunkSpec, ...]:
    return (
        ChunkSpec(
            chunk_id="chunk-000000",
            source_decode_start_pts=0,
            source_decode_end_pts=300,
            output_core_start=0,
            output_core_end=300,
            context_before=0,
            context_after=0,
            scene_ids=(),
            identity_sha256="d" * 64,
        ),
    )


def benchmark() -> BenchmarkResult:
    return BenchmarkResult(
        frames=300,
        elapsed_seconds=Decimal("30"),
        end_to_end_fps=Decimal("10"),
        peak_vram_bytes=2_000_000_000,
        peak_disk_bytes=3_000_000_000,
        worker_count=1,
        variance=Decimal("0.05"),
    )

