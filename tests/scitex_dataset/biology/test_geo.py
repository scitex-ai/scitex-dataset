#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.biology.geo (no network calls)."""

import pytest

from scitex_dataset.biology import geo


class TestGeoExports:
    def test_api_url_constant(self):
        assert geo.GEO_API.startswith("https://eutils.ncbi.nlm.nih.gov/")

    def test_public_callables_present(self):
        for name in ("fetch_datasets", "fetch_all_datasets", "format_dataset"):
            assert hasattr(geo, name), f"missing public name: {name}"
            assert callable(getattr(geo, name))

    def test_all_names_match_module(self):
        for name in geo.__all__:
            assert hasattr(geo, name), f"__all__ lists missing name: {name}"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
