from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engvit.config import AppConfig, JobConfig
from engvit.types import Rational


def make_app_config(roots: tuple[Path, Path, Path, Path]) -> AppConfig:
    input_root, weight_root, wheel_root, output_root = roots
    return AppConfig(
        input_roots=(input_root,),
        weight_roots=(weight_root,),
        wheel_roots=(wheel_root,),
        output_root=output_root,
    )


def test_app_config_rejects_relative_security_root(
    absolute_roots: tuple[Path, Path, Path, Path],
) -> None:
    """Catches resolving an attacker-controlled relative root at runtime."""
    _, weight_root, wheel_root, output_root = absolute_roots
    with pytest.raises(ValidationError, match="absolute"):
        AppConfig(
            input_roots=(Path("relative/input"),),
            weight_roots=(weight_root,),
            wheel_roots=(wheel_root,),
            output_root=output_root,
        )


def test_job_config_forbids_unknown_and_hdr_conversion_fields() -> None:
    """Catches silently accepting an unimplemented HDR or misspelled option."""
    with pytest.raises(ValidationError):
        JobConfig.model_validate(
            {
                "selected_video_index": 0,
                "target_width": 3840,
                "target_height": 2160,
                "hdr_policy": "tone_map",
                "allow_hdr_to_sdr": True,
            }
        )


def test_job_config_requires_target_fps_for_normalization() -> None:
    """Catches admitting VFR normalization without a rational output rate."""
    with pytest.raises(ValidationError, match="target_fps"):
        JobConfig(
            selected_video_index=0,
            target_width=3840,
            target_height=2160,
            fps_policy="normalize_cfr",
        )


def test_job_config_rejects_target_above_app_capability(
    absolute_roots: tuple[Path, Path, Path, Path],
) -> None:
    """Catches bypassing application output limits through a valid job object."""
    app = make_app_config(absolute_roots)
    job = JobConfig(
        selected_video_index=0,
        target_width=8192,
        target_height=4320,
        fps_policy="source_cfr",
    )
    with pytest.raises(ValueError, match="target_width"):
        job.validate_against(app)


def test_job_config_accepts_exact_ntsc_normalization() -> None:
    job = JobConfig(
        selected_video_index=1,
        target_width=3840,
        target_height=2160,
        fps_policy="normalize_cfr",
        target_fps=Rational(30000, 1001),
    )
    assert job.target_fps == Rational(30000, 1001)

