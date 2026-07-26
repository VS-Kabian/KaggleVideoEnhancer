from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from engvit.models.dni import SafeTensorSet, merge_dni


def test_dni_endpoints_and_midpoint_use_fp32_cpu_math() -> None:
    general = SafeTensorSet(
        tensors={
            "weight": np.array([0.0, 2.0], dtype=np.float16),
            "counter": np.array([3], dtype=np.int64),
        }
    )
    weak = SafeTensorSet(
        tensors={
            "weight": np.array([2.0, 4.0], dtype=np.float16),
            "counter": np.array([3], dtype=np.int64),
        }
    )
    middle = merge_dni(general, weak, Decimal("0.5"))
    assert middle.tensors["weight"].dtype == np.float32
    np.testing.assert_array_equal(
        middle.tensors["weight"],
        np.array([1.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(middle.tensors["counter"], np.array([3]))


def test_dni_rejects_key_shape_dtype_and_nonfloat_mismatch() -> None:
    one = SafeTensorSet(tensors={"x": np.array([1], dtype=np.int64)})
    two = SafeTensorSet(tensors={"x": np.array([2], dtype=np.int64)})
    with pytest.raises(ValueError, match="non-floating"):
        merge_dni(one, two, Decimal("0.5"))

