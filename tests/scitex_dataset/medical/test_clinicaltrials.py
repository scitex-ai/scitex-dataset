#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.medical.clinicaltrials (no network calls)."""

import pytest

from scitex_dataset.medical import clinicaltrials


class TestClinicalTrialsExports:
    def test_api_url_constant(self):
        assert clinicaltrials.CLINICALTRIALS_API.startswith(
            "https://clinicaltrials.gov/"
        )

    def test_public_callables_present(self):
        for name in ("fetch_datasets", "fetch_all_datasets", "format_dataset"):
            assert hasattr(clinicaltrials, name)
            assert callable(getattr(clinicaltrials, name))

    def test_all_names_match_module(self):
        for name in clinicaltrials.__all__:
            assert hasattr(clinicaltrials, name)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
