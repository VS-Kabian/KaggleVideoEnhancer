from __future__ import annotations

import hashlib
from pathlib import Path

import imageio_ffmpeg

from engvit.environment import PackageVersion, capture_environment, compare_critical_runtime


def test_capture_environment_records_packages_and_python() -> None:
    """Catches an environment lock that cannot identify its Python/package ABI."""
    lock = capture_environment()
    assert lock.python_version
    assert lock.python_executable_sha256
    assert any(package.name.lower() == "pydantic" for package in lock.packages)
    assert lock.observed_at.endswith("Z")


def test_capture_environment_records_real_ffmpeg_binary() -> None:
    """Catches trusting an encoder name without recording the actual binary."""
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    lock = capture_environment(ffmpeg_path=executable)
    assert lock.ffmpeg is not None
    assert lock.ffmpeg.sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert "ffmpeg version 7.1" in lock.ffmpeg.version
    assert "--enable" in lock.ffmpeg.buildconf
    assert lock.ffmpeg.license_text


def test_compare_critical_runtime_detects_numpy_or_torch_replacement() -> None:
    """Catches a dependency install silently replacing Kaggle's compute stack."""
    before = capture_environment()
    after_packages = tuple(
        PackageVersion(name=package.name, version="999.0")
        if package.name.lower() == "numpy"
        else package
        for package in before.packages
    )
    after = before.model_copy(update={"packages": after_packages, "numpy_version": "999.0"})
    report = compare_critical_runtime(before, after)
    assert report.passed is False
    assert f"numpy_version:{before.numpy_version}->999.0" in report.changes
