"""Strict user and application configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engvit.errors import ConfigurationError
from engvit.types import ContainerPolicy, FPSPolicy, HDRPolicy, Rational


def _require_absolute_roots(paths: tuple[Path, ...], label: str) -> tuple[Path, ...]:
    if not paths:
        raise ValueError(f"{label} must contain at least one approved root")
    resolved: list[Path] = []
    for path in paths:
        if not path.is_absolute():
            raise ValueError(f"{label} entries must be absolute")
        resolved.append(path.resolve(strict=False))
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must not contain duplicate roots")
    return tuple(resolved)


class AppConfig(BaseModel):
    """Administrator-controlled safety limits and approved roots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    input_roots: tuple[Path, ...]
    weight_roots: tuple[Path, ...]
    wheel_roots: tuple[Path, ...]
    output_root: Path
    max_target_width: int = Field(default=7680, ge=2, le=16384)
    max_target_height: int = Field(default=4320, ge=2, le=16384)
    max_duration_seconds: int = Field(default=900, ge=1, le=86400)
    max_output_fps: Rational = Rational(60, 1)
    disk_safety_fraction: float = Field(default=0.20, ge=0.05, le=0.50)
    vram_safety_fraction: float = Field(default=0.15, ge=0.05, le=0.50)

    @field_validator("input_roots", "weight_roots", "wheel_roots")
    @classmethod
    def validate_roots(
        cls, value: tuple[Path, ...], info: object
    ) -> tuple[Path, ...]:
        label = getattr(info, "field_name", "roots")
        return _require_absolute_roots(value, str(label))

    @field_validator("output_root")
    @classmethod
    def validate_output_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("output_root must be absolute")
        return value.resolve(strict=False)

    @model_validator(mode="after")
    def roots_are_separate(self) -> AppConfig:
        read_roots = self.input_roots + self.weight_roots + self.wheel_roots
        if self.output_root in read_roots:
            raise ValueError("output_root must not alias a read-only root")
        return self


class JobConfig(BaseModel):
    """A user's bounded, non-executable job choices."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: str = "1"
    selected_video_index: int = Field(ge=0)
    target_width: int = Field(ge=2, le=16384)
    target_height: int = Field(ge=2, le=16384)
    preset: Literal["conservative", "balanced", "detail"] = "balanced"
    container_policy: ContainerPolicy = "mp4_compatibility"
    hdr_policy: HDRPolicy = "reject"
    fps_policy: FPSPolicy = "source_cfr"
    target_fps: Rational | None = None
    model_id: str | None = None
    denoise_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    experimental_consent: bool = False

    @model_validator(mode="after")
    def fps_choice_is_complete(self) -> JobConfig:
        requires_target = self.fps_policy in ("normalize_cfr", "rife")
        if requires_target and self.target_fps is None:
            raise ValueError(f"target_fps is required for {self.fps_policy}")
        if not requires_target and self.target_fps is not None:
            raise ValueError("target_fps is only valid for normalization or RIFE")
        if self.target_fps is not None and self.target_fps.numerator <= 0:
            raise ValueError("target_fps must be positive")
        return self

    def validate_against(self, app: AppConfig) -> JobConfig:
        if self.target_width > app.max_target_width:
            raise ConfigurationError(
                f"target_width {self.target_width} exceeds {app.max_target_width}"
            )
        if self.target_height > app.max_target_height:
            raise ConfigurationError(
                f"target_height {self.target_height} exceeds {app.max_target_height}"
            )
        if (
            self.target_fps is not None
            and self.target_fps.as_float() > app.max_output_fps.as_float()
        ):
            raise ConfigurationError(
                f"target_fps exceeds {app.max_output_fps.numerator}/"
                f"{app.max_output_fps.denominator}"
            )
        return self
