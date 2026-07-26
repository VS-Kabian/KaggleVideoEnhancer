"""Capture immutable evidence about the active execution environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PackageVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str


class BinaryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str
    version: str
    license_text: str
    buildconf: str


class GPUInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    name: str
    driver_version: str
    total_memory_mib: int
    compute_capability: str | None


class EnvironmentLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    observed_at: str
    python_version: str
    python_implementation: str
    python_executable: Path
    python_executable_sha256: str
    platform: str
    packages: tuple[PackageVersion, ...]
    torch_version: str | None
    torchvision_version: str | None
    numpy_version: str | None
    ffmpeg: BinaryEvidence | None
    gpus: tuple[GPUInfo, ...]


class EnvironmentDrift(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    changes: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(executable: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable.name} {' '.join(arguments)} failed: {output.strip()}"
        )
    return output.strip()


def _capture_ffmpeg(path: Path) -> BinaryEvidence:
    resolved = path.resolve(strict=True)
    version_output = _run(resolved, "-version")
    version = version_output.splitlines()[0] if version_output else ""
    return BinaryEvidence(
        path=resolved,
        sha256=_sha256(resolved),
        version=version,
        license_text=_run(resolved, "-L"),
        buildconf=_run(resolved, "-buildconf"),
    )


def _package_snapshot() -> tuple[PackageVersion, ...]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return tuple(
        PackageVersion(name=name, version=version)
        for name, version in sorted(packages.items(), key=lambda item: item[0].lower())
    )


def _version(packages: tuple[PackageVersion, ...], name: str) -> str | None:
    lowered = name.lower()
    return next(
        (package.version for package in packages if package.name.lower() == lowered),
        None,
    )


def _capture_gpus() -> tuple[GPUInfo, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    query = (
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        output = _run(Path(executable), *query)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return ()
    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    driver_version=parts[2],
                    total_memory_mib=int(parts[3]),
                    compute_capability=parts[4] or None,
                )
            )
        except ValueError:
            continue
    return tuple(gpus)


def capture_environment(*, ffmpeg_path: Path | None = None) -> EnvironmentLock:
    """Capture runtime evidence without importing or mutating Torch/CUDA."""
    executable = Path(sys.executable).resolve(strict=True)
    packages = _package_snapshot()
    return EnvironmentLock(
        observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_executable=executable,
        python_executable_sha256=_sha256(executable),
        platform=platform.platform(),
        packages=packages,
        torch_version=_version(packages, "torch"),
        torchvision_version=_version(packages, "torchvision"),
        numpy_version=_version(packages, "numpy"),
        ffmpeg=_capture_ffmpeg(ffmpeg_path) if ffmpeg_path is not None else None,
        gpus=_capture_gpus(),
    )


def compare_critical_runtime(
    before: EnvironmentLock,
    after: EnvironmentLock,
) -> EnvironmentDrift:
    """Detect replacement of the Python/Torch/CUDA/NumPy execution stack."""
    changes: list[str] = []
    scalar_fields = (
        "python_version",
        "python_executable_sha256",
        "torch_version",
        "torchvision_version",
        "numpy_version",
    )
    for field_name in scalar_fields:
        old = getattr(before, field_name)
        new = getattr(after, field_name)
        if old != new:
            changes.append(f"{field_name}:{old}->{new}")

    def critical_packages(lock: EnvironmentLock) -> dict[str, str]:
        return {
            package.name.lower(): package.version
            for package in lock.packages
            if package.name.lower().startswith(("cuda", "nvidia-", "triton"))
        }

    old_packages = critical_packages(before)
    new_packages = critical_packages(after)
    for name in sorted(set(old_packages) | set(new_packages)):
        old = old_packages.get(name)
        new = new_packages.get(name)
        if old != new:
            changes.append(f"package:{name}:{old}->{new}")
    return EnvironmentDrift(passed=not changes, changes=tuple(changes))
