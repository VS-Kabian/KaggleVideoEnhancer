"""Hash-, license-, signature-, and environment-gated model resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from engvit.environment import EnvironmentLock
from engvit.licenses import IntendedUse, LicenseRegistry
from engvit.models.realesrgan import validate_realesrgan_signature
from engvit.models.weights import inspect_safetensors
from engvit.supply_chain import require_safe_runtime_weight


class RequiredTensor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dtype: str
    shape: tuple[int, ...]


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    enabled: bool
    architecture: str
    scale: int = Field(ge=1, le=8)
    input_channels: int = Field(ge=1, le=4)
    output_channels: int = Field(ge=1, le=4)
    weight_filename: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    weight_bytes: int | None = Field(default=None, ge=1)
    code_ref: str
    code_license: str
    required_packages: dict[str, tuple[str, ...]]
    precision: tuple[Literal["fp32", "fp16"], ...]
    required_tensors: tuple[RequiredTensor, ...]

    @model_validator(mode="after")
    def enabled_entry_is_complete(self) -> ModelEntry:
        if self.enabled and (
            self.weight_filename is None
            or self.weight_sha256 is None
            or self.weight_bytes is None
            or not self.required_tensors
        ):
            raise ValueError("enabled model entry has incomplete weight/signature data")
        return self


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    models: tuple[ModelEntry, ...]


class ModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    architecture: str
    scale: int
    input_channels: int
    output_channels: int
    weight_path: Path
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_bytes: int
    code_ref: str
    code_license: str
    precision: tuple[Literal["fp32", "fp16"], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(
        self,
        entries: tuple[ModelEntry, ...],
        licenses: LicenseRegistry,
        weight_roots: tuple[Path, ...],
    ) -> None:
        by_id = {entry.model_id: entry for entry in entries}
        if len(by_id) != len(entries):
            raise ValueError("model registry contains duplicate IDs")
        self._entries = by_id
        self._licenses = licenses
        self._weight_roots = weight_roots

    @classmethod
    def from_files(
        cls,
        registry_path: Path,
        licenses: LicenseRegistry,
        *,
        weight_roots: tuple[Path, ...],
    ) -> ModelRegistry:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        registry = TypeAdapter(RegistryFile).validate_python(payload)
        return cls(registry.models, licenses, weight_roots)

    def resolve(
        self,
        model_id: str,
        environment: EnvironmentLock,
        intended_use: IntendedUse,
    ) -> ModelArtifact:
        entry = self._entries.get(model_id)
        if entry is None:
            raise ValueError(f"unknown model ID: {model_id}")
        if not entry.enabled:
            raise ValueError(f"model {model_id} is disabled pending verified evidence")
        license_record = self._licenses.require_model(model_id, intended_use)
        package_versions = {
            package.name.casefold(): package.version for package in environment.packages
        }
        for package, allowed_versions in entry.required_packages.items():
            observed = package_versions.get(package.casefold())
            if observed not in allowed_versions:
                raise ValueError(
                    f"model {model_id} requires audited {package} version; "
                    f"observed {observed}"
                )
        assert entry.weight_filename is not None
        assert entry.weight_sha256 is not None
        assert entry.weight_bytes is not None
        candidates = tuple(
            root / entry.weight_filename for root in self._weight_roots
        )
        existing = next((path for path in candidates if path.is_file()), None)
        if existing is None:
            raise ValueError(f"weight file is missing for {model_id}")
        matching_root = next(
            root for root in self._weight_roots if existing.is_relative_to(root)
        )
        weight = require_safe_runtime_weight(existing, matching_root)
        if weight.stat().st_size != entry.weight_bytes:
            raise ValueError(f"weight size mismatch for {model_id}")
        digest = _sha256(weight)
        if digest != entry.weight_sha256:
            raise ValueError(f"weight hash mismatch for {model_id}")
        if license_record.weight_sha256 != digest:
            raise ValueError(f"license evidence hash mismatch for {model_id}")
        header = inspect_safetensors(weight)
        validate_realesrgan_signature(
            header,
            tuple(
                (tensor.name, tensor.dtype, tensor.shape)
                for tensor in entry.required_tensors
            ),
        )
        return ModelArtifact(
            model_id=entry.model_id,
            architecture=entry.architecture,
            scale=entry.scale,
            input_channels=entry.input_channels,
            output_channels=entry.output_channels,
            weight_path=weight,
            weight_sha256=digest,
            weight_bytes=entry.weight_bytes,
            code_ref=entry.code_ref,
            code_license=entry.code_license,
            precision=entry.precision,
        )

