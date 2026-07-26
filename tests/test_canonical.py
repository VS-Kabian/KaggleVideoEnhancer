from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from engvit.canonical import canonical_bytes, canonical_sha256
from engvit.types import Rational


def test_canonical_bytes_are_sorted_and_tag_non_json_types() -> None:
    """Catches hashes changing with dictionary order or lossy type conversion."""
    value = {
        "z": Rational(1, 2),
        "a": Decimal("1.20"),
        "path": Path("folder") / "file.bin",
    }
    assert canonical_bytes(value, projection="full") == (
        b'{"a":{"__decimal__":"1.20"},'
        b'"path":{"__path__":"folder/file.bin"},'
        b'"z":{"__rational__":[1,2]}}'
    )


def test_canonical_hash_ignores_dictionary_insertion_order() -> None:
    left = {"model": "x2", "scale": 2}
    right = {"scale": 2, "model": "x2"}
    assert canonical_sha256(left, projection="full") == canonical_sha256(
        right, projection="full"
    )


def test_identity_projection_excludes_only_declared_volatile_fields() -> None:
    """Catches progress observations invalidating an otherwise identical plan."""
    left = {
        "source_sha256": "abc",
        "progress": {"frames": 10},
        "observed_at": "first",
        "recipe": {"model": "x2"},
    }
    right = {
        "source_sha256": "abc",
        "progress": {"frames": 999},
        "observed_at": "second",
        "recipe": {"model": "x2"},
    }
    assert canonical_sha256(left, projection="identity") == canonical_sha256(
        right, projection="identity"
    )


def test_canonicalization_rejects_non_finite_float() -> None:
    """Catches serializing a value that canonical JSON cannot represent."""
    with pytest.raises(ValueError, match="finite"):
        canonical_bytes({"bad": float("nan")}, projection="full")


def test_unknown_projection_is_rejected() -> None:
    with pytest.raises(ValueError, match="projection"):
        canonical_bytes({"value": 1}, projection="guess")

