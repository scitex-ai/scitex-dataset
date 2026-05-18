#!/usr/bin/env python3
"""Smoke tests for scitex_dataset.medical.clinicaltrials (no network calls)."""

import pytest

from scitex_dataset.medical import clinicaltrials


class TestClinicalTrialsExports:
    def test_api_constant_starts_with_clinicaltrials_gov_host(self):
        # Arrange
        expected_prefix = "https://clinicaltrials.gov/"
        # Act
        actual = clinicaltrials.CLINICALTRIALS_API
        # Assert
        assert actual.startswith(expected_prefix)

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_callable_attribute_present_on_clinicaltrials_module(self, name):
        # Arrange
        module = clinicaltrials
        # Act
        present = hasattr(module, name)
        # Assert
        assert present, f"missing public name: {name}"

    @pytest.mark.parametrize(
        "name", ["fetch_datasets", "fetch_all_datasets", "format_dataset"]
    )
    def test_public_attribute_is_callable_on_clinicaltrials_module(self, name):
        # Arrange
        module = clinicaltrials
        # Act
        attr = getattr(module, name, None)
        # Assert
        assert callable(attr)

    def test_clinicaltrials_all_list_references_only_existing_names(self):
        # Arrange
        names = list(clinicaltrials.__all__)
        # Act
        missing = [n for n in names if not hasattr(clinicaltrials, n)]
        # Assert
        assert missing == [], f"__all__ lists missing names: {missing}"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
