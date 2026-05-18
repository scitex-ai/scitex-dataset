#!/usr/bin/env python3
"""Tests for scitex_dataset._cli._introspect helpers."""

import pytest

from scitex_dataset._cli._introspect import (
    TYPE_COLORS,
    _format_python_signature,
)


class TestTypeColors:
    def test_known_type_colors_present_in_mapping(self):
        # Arrange
        expected = {"M", "C", "F", "V"}
        # Act
        actual_keys = set(TYPE_COLORS.keys())
        # Assert
        assert expected <= actual_keys

    def test_all_color_values_are_non_empty_strings(self):
        # Arrange
        values = list(TYPE_COLORS.values())
        # Act
        all_ok = all(isinstance(v, str) and v for v in values)
        # Assert
        assert all_ok


class TestFormatPythonSignature:
    def test_returns_tuple_containing_sample_name(self):
        # Arrange
        def sample(x: int = 1, y: str = "z") -> None:
            return None

        # Act
        name_str, _sig_str = _format_python_signature(sample)
        # Assert
        assert "sample" in name_str

    def test_returns_two_strings_for_signed_callable(self):
        # Arrange
        def sample(x: int = 1, y: str = "z") -> None:
            return None

        # Act
        name_str, sig_str = _format_python_signature(sample)
        # Assert
        assert isinstance(name_str, str) and isinstance(sig_str, str)

    def test_returns_tuple_of_length_two_for_builtin(self):
        # Arrange
        # Built-ins like `len` may or may not have inspect.signature support;
        # the helper must not raise either way.
        # Act
        out = _format_python_signature(len)
        # Assert
        assert isinstance(out, tuple) and len(out) == 2


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
