from __future__ import annotations

import pytest

from engvit.types import Rational


def test_rational_normalizes_sign_and_common_factor() -> None:
    """Catches storing equivalent rates under different identities."""
    assert Rational(60000, -2002) == Rational(-30000, 1001)


def test_rational_rejects_zero_denominator() -> None:
    """Catches accepting a rate that cannot be used in timing arithmetic."""
    with pytest.raises(ValueError, match="denominator"):
        Rational(1, 0)

