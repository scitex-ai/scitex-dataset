#!/usr/bin/env python3
"""Tests for scitex_dataset._cli._introspect helpers."""


import pytest

from scitex_dataset._cli._introspect import (
    TYPE_COLORS,
    _format_python_signature,
)


class TestTypeColors:
    def test_known_type_colors_present(self):
        # Module / Class / Function / Variable colors.
        assert {"M", "C", "F", "V"} <= set(TYPE_COLORS.keys())

    def test_colors_are_strings(self):
        for v in TYPE_COLORS.values():
            assert isinstance(v, str) and v


class TestFormatPythonSignature:
    def test_returns_two_strings_for_callable(self):
        def sample(x: int = 1, y: str = "z") -> None:
            return None

        name_str, sig_str = _format_python_signature(sample)
        assert isinstance(name_str, str)
        assert isinstance(sig_str, str)
        assert "sample" in name_str

    def test_handles_unsignable_callable(self):
        # Built-ins like `len` may or may not have inspect.signature
        # support; the helper must not raise either way.
        out = _format_python_signature(len)
        assert isinstance(out, tuple) and len(out) == 2


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
