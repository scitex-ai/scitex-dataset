#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 23:58:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_physionet.py

"""Tests for PhysioNet dataset fetcher."""

from unittest.mock import MagicMock

from scitex_dataset.neuroscience.physionet import PHYSIONET_API, format_dataset


def test_physionet_api_url():
    """Test that API URL is correct."""
    assert PHYSIONET_API == "https://physionet.org"


def test_format_dataset(physionet_database):
    """Test formatting a PhysioNet database."""
    formatted = format_dataset(physionet_database)

    assert formatted["id"] == "sample-eeg-db"
    assert formatted["name"] == "Sample EEG Database"
    assert formatted["version"] == "1.0.0"
    assert "epilepsy" in formatted["abstract"].lower()
    assert formatted["doi"] == "10.13026/xxxx-yyyy"
    assert "Attribution" in formatted["license"]
    assert formatted["n_subjects"] == 100
    assert formatted["n_records"] == 500
    assert formatted["size_gb"] == 20.0
    assert formatted["publish_date"] == "2023-06-15"
    assert "physionet.org/content/sample-eeg-db" in formatted["url"]
    assert formatted["data_access"] == "open"


def test_format_dataset_string_license():
    """Test formatting with string license (not dict)."""
    database = {
        "slug": "test-db",
        "title": "Test Database",
        "license": "ODC-By v1.0",
    }

    formatted = format_dataset(database)

    assert formatted["license"] == "ODC-By v1.0"


def test_format_dataset_minimal():
    """Test formatting with minimal fields."""
    database = {}

    formatted = format_dataset(database)

    assert formatted["id"] == ""
    assert formatted["name"] == "N/A"
    assert formatted["n_subjects"] == 0
    assert formatted["n_records"] == 0
    assert formatted["size_gb"] == 0.0
    assert formatted["url"] == ""


def test_format_dataset_alternate_keys():
    """Test formatting with alternate key names."""
    database = {
        "short_name": "alt-db",
        "name": "Alternate Database",
        "description": "An alternate description.",
    }

    formatted = format_dataset(database)

    assert formatted["id"] == "alt-db"
    assert formatted["name"] == "Alternate Database"
    assert formatted["abstract"] == "An alternate description."


def test_fetch_datasets_mocked(mock_httpx_get):
    """Test fetch_datasets with mocked HTTP."""
    from scitex_dataset.neuroscience.physionet import fetch_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"slug": "db1", "title": "Database 1"},
        {"slug": "db2", "title": "Database 2"},
    ]
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    result = fetch_datasets(page=1)

    # PhysioNet returns list directly
    assert isinstance(result, list)
    assert len(result) == 2


def test_fetch_all_datasets_list_response(mock_httpx_get):
    """Test fetch_all_datasets with list response (no pagination)."""
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = [{"slug": f"db{i}"} for i in range(1, 6)]
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    datasets = fetch_all_datasets()

    assert len(datasets) == 5
    assert mock_httpx_get.call_count == 1


def test_fetch_all_datasets_max_limit(mock_httpx_get):
    """Test fetch_all_datasets respects max_datasets."""
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = [{"slug": f"db{i}"} for i in range(1, 11)]
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    datasets = fetch_all_datasets(max_datasets=3)

    assert len(datasets) == 3


def test_fetch_all_datasets_paginated_response(mock_httpx_get):
    """Test fetch_all_datasets with paginated dict response."""
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    page1_response = MagicMock()
    page1_response.json.return_value = {
        "results": [{"slug": f"db{i}"} for i in range(1, 4)],
        "next": "http://next",
    }
    page1_response.raise_for_status = MagicMock()

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "results": [{"slug": f"db{i}"} for i in range(4, 6)],
        "next": None,
    }
    page2_response.raise_for_status = MagicMock()

    mock_httpx_get.side_effect = [page1_response, page2_response]

    datasets = fetch_all_datasets()

    assert len(datasets) == 5


# EOF
