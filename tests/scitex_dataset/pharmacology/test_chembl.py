#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.pharmacology.chembl (no network calls)."""

import pytest

from scitex_dataset.pharmacology import chembl


class TestChEMBLExports:
    def test_chembl_api_constant_starts_with_ebi_chembl_host(self):
        # Arrange
        expected_prefix = "https://www.ebi.ac.uk/chembl/"
        # Act
        actual = chembl.CHEMBL_API
        # Assert
        assert actual.startswith(expected_prefix)

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_callable_attribute_present_on_chembl_module(self, name):
        # Arrange
        module = chembl
        # Act
        present = hasattr(module, name)
        # Assert
        assert present, f"missing public name: {name}"

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_attribute_is_callable_on_chembl_module(self, name):
        # Arrange
        module = chembl
        # Act
        attr = getattr(module, name, None)
        # Assert
        assert callable(attr)

    def test_chembl_all_list_references_only_existing_names(self):
        # Arrange
        names = list(chembl.__all__)
        # Act
        missing = [n for n in names if not hasattr(chembl, n)]
        # Assert
        assert missing == [], f"__all__ lists missing names: {missing}"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
