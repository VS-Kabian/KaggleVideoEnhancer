"""Convert a reviewed legacy tensor checkpoint in a constrained helper process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LEGACY_SUFFIXES = {".pth", ".pt", ".ckpt", ".pkl"}
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_tuple(version: str) -> tuple[int, int]:
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"unparseable PyTorch version: {version}") from exc


def _validate_process() -> None:
    if not (
        os.environ.get("ENGVIT_CONVERSION_ISOLATED") == "1"
        and os.environ.get("ENGVIT_NETWORK_DISABLED") == "1"
    ):
        raise PermissionError(
            "checkpoint conversion requires an isolated, network-disabled process"
        )
    exposed = [
        name
        for name, value in os.environ.items()
        if value and any(marker in name.upper() for marker in _SECRET_MARKERS)
    ]
    if exposed:
        raise PermissionError(
            "checkpoint conversion environment contains secret-bearing variables: "
            + ",".join(sorted(exposed))
        )


def _unwrap_state_dict(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint root must be a tensor mapping")
    for key in ("params_ema", "params", "state_dict"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
            break
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint does not contain a tensor mapping")
    return value


def _convert(
    source: Path,
    destination: Path,
    inventory_path: Path,
    *,
    max_tensors: int,
    max_elements: int,
) -> None:
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    torch_version = importlib.metadata.version("torch")
    if _version_tuple(torch_version) < (2, 6):
        raise RuntimeError("checkpoint conversion requires PyTorch 2.6 or newer")

    import torch
    from safetensors.torch import save_file

    loaded = torch.load(source, map_location="cpu", weights_only=True)
    mapping = _unwrap_state_dict(loaded)
    if len(mapping) > max_tensors:
        raise ValueError(f"checkpoint exceeds max_tensors={max_tensors}")

    tensors: dict[str, Any] = {}
    inventory: list[dict[str, object]] = []
    total_elements = 0
    for name, value in sorted(mapping.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("checkpoint must contain string-to-tensor entries only")
        if value.is_sparse:
            raise ValueError(f"sparse tensor is not allowed: {name}")
        total_elements += int(value.numel())
        if total_elements > max_elements:
            raise ValueError(f"checkpoint exceeds max_elements={max_elements}")
        tensor = value.detach().to(device="cpu").contiguous()
        tensors[name] = tensor
        inventory.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "elements": int(tensor.numel()),
            }
        )

    partial = destination.with_name(f".{destination.name}.partial")
    save_file(tensors, str(partial), metadata={"format": "engvit-tensors-v1"})
    partial.replace(destination)
    evidence = {
        "schema_version": "1",
        "torch_version": torch_version,
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
        "tensor_count": len(tensors),
        "total_elements": total_elements,
        "tensors": inventory,
    }
    inventory_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--max-input-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-tensors", type=int, default=100_000)
    parser.add_argument("--max-elements", type=int, default=4_000_000_000)
    args = parser.parse_args(argv)

    try:
        _validate_process()
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    source = args.input.resolve(strict=True)
    destination = args.output.resolve(strict=False)
    inventory = (
        args.inventory.resolve(strict=False)
        if args.inventory
        else destination.with_suffix(".inventory.json")
    )
    if source.suffix.lower() not in _LEGACY_SUFFIXES:
        print("input must be a reviewed legacy checkpoint", file=sys.stderr)
        return 4
    if destination.suffix.lower() != ".safetensors":
        print("output must use .safetensors", file=sys.stderr)
        return 4
    if source.stat().st_size > args.max_input_bytes:
        print("input exceeds max-input-bytes", file=sys.stderr)
        return 4
    if destination.exists() or inventory.exists():
        print("output or inventory already exists", file=sys.stderr)
        return 4
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        _convert(
            source,
            destination,
            inventory,
            max_tensors=args.max_tensors,
            max_elements=args.max_elements,
        )
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
