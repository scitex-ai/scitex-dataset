#!/usr/bin/env python3
"""Tests for ai_for_science._sigfig (sig-fig-aware float comparison).

The numeric family's n==1 / degenerate-PI fallback fires on real data,
so every branch of ``sigfigs`` and ``is_close_sigfig`` is exercised with
real values (no mocks).
"""

import pytest

from scitex_dataset.ai_for_science._sigfig import is_close_sigfig, sigfigs


class TestSigfigsBranches:
    def test_empty_string_is_zero(self):
        # Arrange
        value = ""
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 0

    def test_sign_is_stripped(self):
        # Arrange — leading minus must not count as a figure.
        value = "-0.59375"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 5

    def test_plus_sign_is_stripped(self):
        # Arrange
        value = "+12.5"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 3

    def test_bare_sign_only_is_zero(self):
        # Arrange — only a sign, nothing after it.
        value = "-"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 0

    def test_scientific_notation_counts_mantissa(self):
        # Arrange
        value = "1.9E-05"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 2

    def test_scientific_notation_lower_e(self):
        # Arrange
        value = "1.00e3"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 3

    def test_zero_is_one(self):
        # Arrange
        value = 0
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 1

    def test_decimal_with_integer_part(self):
        # Arrange
        value = "12.34"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 4

    def test_pure_fractional_strips_leading_zeros(self):
        # Arrange
        value = "0.0059375"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 5

    def test_integer_trailing_zeros_are_ambiguous(self):
        # Arrange — "100" trailing zeros conservatively count as 1 SF.
        value = "100"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 1

    def test_plain_integer_counts_digits(self):
        # Arrange
        value = "305"
        # Act
        n = sigfigs(value)
        # Assert
        assert n == 3


class TestIsCloseSigfig:
    def test_agrees_at_published_precision(self):
        # Arrange
        reference = 0.9996
        # Act
        ok = is_close_sigfig(0.99956, reference)
        # Assert
        assert ok is True

    def test_precision_loss_is_rejected(self):
        # Arrange — 0.59 loses precision vs the 5-SF reference.
        reference = "0.59375"
        # Act
        ok = is_close_sigfig(0.59, reference)
        # Assert
        assert ok is False

    def test_zero_reference_uses_abs_tol(self):
        # Arrange — b == 0 falls back to the absolute tolerance.
        reference = 0
        # Act
        ok = is_close_sigfig(1e-12, reference)
        # Assert
        assert ok is True

    def test_zero_reference_rejects_far_value(self):
        # Arrange
        reference = 0
        # Act
        ok = is_close_sigfig(1.0, reference)
        # Assert
        assert ok is False


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
