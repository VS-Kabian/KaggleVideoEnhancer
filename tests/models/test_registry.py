from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from engvit.environment import capture_environment
from engvit.licenses import IntendedUse, LicenseRegistry
from engvit.models.registry import ModelRegistry
from tests.models.helpers import write_safetensors


def test_workspace_registry_blocks_unverified_weight_before_importing_torch() -> None:
    registry = ModelRegistry.from_files(
        Path("models/registry.yaml"),
        LicenseRegistry.from_yaml(Path("licenses/model-weights.yaml")),
        weight_roots=(Path("weights").resolve(),),
    )
    with pytest.raises(ValueError, match=r"disabled|unverified"):
        registry.resolve(
            "realesrgan-x4plus",
            capture_environment(),
            IntendedUse(private=True, commercial=False, redistribute=False),
        )


def test_verified_registry_resolves_only_exact_hash_and_signature(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    model = weights / "model.safetensors"
    digest = write_safetensors(
        model,
        {"conv.weight": ("F32", (1, 1, 1, 1), b"\0\0\0\0")},
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1",
                "models": [
                    {
                        "model_id": "test-x2",
                        "enabled": True,
                        "architecture": "RRDBNet",
                        "scale": 2,
                        "input_channels": 3,
                        "output_channels": 3,
                        "weight_filename": "model.safetensors",
                        "weight_sha256": digest,
                        "weight_bytes": model.stat().st_size,
                        "code_ref": "spandrel-test-ref",
                        "code_license": "MIT",
                        "required_packages": {},
                        "precision": ["fp32"],
                        "required_tensors": [
                            {
                                "name": "conv.weight",
                                "dtype": "F32",
                                "shape": [1, 1, 1, 1],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    license_path = tmp_path / "licenses.yaml"
    license_path.write_text(
        yaml.safe_dump(
            [
                {
                    "model_id": "test-x2",
                    "code_license": "MIT",
                    "code_url": "https://example.invalid/code",
                    "weight_status": "verified",
                    "weight_terms_url": "https://example.invalid/terms",
                    "weight_sha256": digest,
                    "reviewed_by": "test",
                    "reviewed_on": "2026-07-26",
                    "private_use": True,
                    "commercial_use": False,
                    "redistribution": False,
                    "notices": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry.from_files(
        registry_path,
        LicenseRegistry.from_yaml(license_path),
        weight_roots=(weights,),
    )
    artifact = registry.resolve(
        "test-x2",
        capture_environment(),
        IntendedUse(private=True, commercial=False, redistribute=False),
    )
    assert artifact.weight_sha256 == hashlib.sha256(model.read_bytes()).hexdigest()
    assert artifact.scale == 2
    model.write_bytes(model.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="hash"):
        registry.resolve(
            "test-x2",
            capture_environment(),
            IntendedUse(private=True, commercial=False, redistribute=False),
        )
