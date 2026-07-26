"""Canonical JSON and SHA-256 identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from engvit.errors import CanonicalizationError
from engvit.types import Rational

_PROJECTIONS = frozenset({"full", "identity"})
_IDENTITY_EXCLUDED_FIELDS = frozenset(
    {
        "absolute_path",
        "created_at",
        "elapsed",
        "eta",
        "host_path",
        "observed_at",
        "progress",
        "runtime",
        "timestamp",
        "updated_at",
    }
)


def _project_mapping(value: Mapping[str, Any], projection: str) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CanonicalizationError("canonical mappings require string keys")
        if projection == "identity" and key in _IDENTITY_EXCLUDED_FIELDS:
            continue
        projected[key] = _normalize(item, projection)
    return projected


def _normalize(value: Any, projection: str) -> Any:
    if isinstance(value, Rational):
        return {"__rational__": [value.numerator, value.denominator]}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalizationError("Decimal values must be finite")
        return {"__decimal__": str(value)}
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, Enum):
        return _normalize(value.value, projection)
    if isinstance(value, BaseModel):
        return _project_mapping(value.model_dump(mode="python"), projection)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
        return _project_mapping(fields, projection)
    if isinstance(value, Mapping):
        return _project_mapping(value, projection)
    if isinstance(value, tuple):
        return [_normalize(item, projection) for item in value]
    if isinstance(value, list):
        return [_normalize(item, projection) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("float values must be finite")
        return value
    if isinstance(value, Sequence):
        return [_normalize(item, projection) for item in value]
    raise CanonicalizationError(
        f"unsupported canonical type: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_bytes(value: object, *, projection: str) -> bytes:
    """Serialize a value under a named, stable identity projection."""
    if projection not in _PROJECTIONS:
        raise ValueError(f"unknown canonical projection: {projection}")
    normalized = _normalize(value, projection)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(str(exc)) from exc
    return text.encode("utf-8")


def canonical_sha256(value: object, *, projection: str) -> str:
    return hashlib.sha256(canonical_bytes(value, projection=projection)).hexdigest()
