#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.biology.geo (no network calls)."""

import pytest

from scitex_dataset.biology import geo


class TestGeoExports:
    def test_geo_api_constant_starts_with_eutils_ncbi_host(self):
        # Arrange
        expected_prefix = "https://eutils.ncbi.nlm.nih.gov/"
        # Act
        actual = geo.GEO_API
        # Assert
        assert actual.startswith(expected_prefix)

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_callable_attribute_present_on_geo_module(self, name):
        # Arrange
        module = geo
        # Act
        present = hasattr(module, name)
        # Assert
        assert present, f"missing public name: {name}"

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_attribute_is_callable_on_geo_module(self, name):
        # Arrange
        module = geo
        # Act
        attr = getattr(module, name, None)
        # Assert
        assert callable(attr)

    def test_geo_module_all_list_references_only_existing_names(self):
        # Arrange
        names = list(geo.__all__)
        # Act
        missing = [n for n in names if not hasattr(geo, n)]
        # Assert
        assert missing == [], f"__all__ lists missing names: {missing}"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
