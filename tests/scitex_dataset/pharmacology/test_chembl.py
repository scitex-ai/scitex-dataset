#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.pharmacology.chembl (no network calls)."""

import pytest

from scitex_dataset.pharmacology import chembl


class TestChEMBLExports:
    def test_api_url_constant(self):
        assert chembl.CHEMBL_API.startswith("https://www.ebi.ac.uk/chembl/")

    def test_public_callables_present(self):
        for name in ("fetch_datasets", "fetch_all_datasets", "format_dataset"):
            assert hasattr(chembl, name)
            assert callable(getattr(chembl, name))

    def test_all_names_match_module(self):
        for name in chembl.__all__:
            assert hasattr(chembl, name)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
