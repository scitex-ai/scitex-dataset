#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-14 09:10:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_zenodo.py

"""Tests for Zenodo dataset fetcher."""

from unittest.mock import MagicMock

from scitex_dataset.general.zenodo import ZENODO_API, format_dataset


def test_zenodo_api_url():
    """Test that API URL is correct."""
    assert ZENODO_API == "https://zenodo.org/api"


def test_format_dataset_full():
    """Test formatting a full Zenodo record."""
    record = {
        "id": 12345,
        "doi": "10.5281/zenodo.12345",
        "created": "2023-01-15T10:00:00Z",
        "updated": "2023-06-01T12:00:00Z",
        "metadata": {
            "title": "Sample Neuroscience Dataset",
            "description": "A dataset for testing.",
            "publication_date": "2023-01-15",
            "version": "1.0.0",
            "creators": [
                {"name": "Smith, John"},
                {"name": "Doe, Jane"},
            ],
            "keywords": ["neuroscience", "eeg"],
            "subjects": [{"term": "brain"}],
            "license": {"id": "cc-by-4.0"},
            "resource_type": {"type": "dataset", "subtype": ""},
        },
        "files": [
            {"size": 1073741824},  # 1 GB
            {"size": 536870912},  # 0.5 GB
        ],
        "stats": {"views": 500, "downloads": 200},
        "links": {"html": "https://zenodo.org/record/12345"},
    }

    formatted = format_dataset(record)

    assert formatted["id"] == "12345"
    assert formatted["doi"] == "10.5281/zenodo.12345"
    assert formatted["name"] == "Sample Neuroscience Dataset"
    assert formatted["description"] == "A dataset for testing."
    assert formatted["created"] == "2023-01-15T10:00:00Z"
    assert formatted["modified"] == "2023-06-01T12:00:00Z"
    assert formatted["publish_date"] == "2023-01-15"
    assert formatted["version"] == "1.0.0"
    assert formatted["authors"] == ["Smith, John", "Doe, Jane"]
    assert "neuroscience" in formatted["keywords"]
    assert "brain" in formatted["keywords"]
    assert formatted["license"] == "cc-by-4.0"
    assert formatted["dataset_type"] == "dataset"
    assert formatted["n_files"] == 2
    assert formatted["size_bytes"] == 1073741824 + 536870912
    assert formatted["size_gb"] > 0
    assert formatted["views"] == 500
    assert formatted["downloads"] == 200
    assert formatted["url"] == "https://zenodo.org/record/12345"
    assert formatted["source"] == "zenodo"


def test_format_dataset_minimal():
    """Test formatting with minimal fields."""
    record = {"id": 99999}

    formatted = format_dataset(record)

    assert formatted["id"] == "99999"
    assert formatted["name"] == ""
    assert formatted["doi"] == ""
    assert formatted["authors"] == []
    assert formatted["keywords"] == []
    assert formatted["n_files"] == 0
    assert formatted["size_bytes"] == 0
    assert formatted["size_gb"] == 0
    assert formatted["views"] == 0
    assert formatted["downloads"] == 0
    assert formatted["source"] == "zenodo"


def test_format_dataset_string_license():
    """Test formatting with string license (not dict)."""
    record = {
        "id": 11111,
        "metadata": {"license": "MIT"},
    }

    formatted = format_dataset(record)

    assert formatted["license"] == "MIT"


def test_format_dataset_string_resource_type():
    """Test formatting with string resource_type (not dict)."""
    record = {
        "id": 22222,
        "metadata": {"resource_type": "dataset"},
    }

    formatted = format_dataset(record)

    assert formatted["dataset_type"] == "dataset"


def test_format_dataset_fallback_url():
    """Test URL fallback when links.html is missing."""
    record = {"id": 33333}

    formatted = format_dataset(record)

    assert formatted["url"] == "https://zenodo.org/record/33333"


def test_fetch_datasets_mocked(mock_httpx_get):
    """Test fetch_datasets with mocked HTTP."""
    from scitex_dataset.general.zenodo import fetch_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hits": {
            "hits": [{"id": 1}, {"id": 2}],
            "total": 2,
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    result = fetch_datasets(query="neuroscience", page=1, size=25)

    assert "hits" in result
    assert len(result["hits"]["hits"]) == 2
    mock_httpx_get.assert_called_once()


def test_fetch_datasets_size_cap(mock_httpx_get):
    """Test that size is capped at 25 for unauthenticated requests."""
    from scitex_dataset.general.zenodo import fetch_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {"hits": {"hits": [], "total": 0}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    fetch_datasets(size=100)

    # Verify URL contains size=25 (capped)
    call_url = mock_httpx_get.call_args[0][0]
    assert "size=25" in call_url


def test_fetch_all_datasets_pagination(mock_httpx_get):
    """Test fetch_all_datasets handles pagination."""
    from scitex_dataset.general.zenodo import fetch_all_datasets

    page1_response = MagicMock()
    page1_response.json.return_value = {
        "hits": {
            "hits": [{"id": i} for i in range(1, 4)],
            "total": 5,
        }
    }
    page1_response.raise_for_status = MagicMock()

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "hits": {
            "hits": [{"id": i} for i in range(4, 6)],
            "total": 5,
        }
    }
    page2_response.raise_for_status = MagicMock()

    mock_httpx_get.side_effect = [page1_response, page2_response]

    datasets = fetch_all_datasets()

    assert len(datasets) == 5
    assert mock_httpx_get.call_count == 2


def test_fetch_all_datasets_max_limit(mock_httpx_get):
    """Test fetch_all_datasets respects max_datasets."""
    from scitex_dataset.general.zenodo import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hits": {
            "hits": [{"id": i} for i in range(1, 11)],
            "total": 100,
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    datasets = fetch_all_datasets(max_datasets=3)

    assert len(datasets) == 3


def test_fetch_all_datasets_empty(mock_httpx_get):
    """Test fetch_all_datasets with empty response."""
    from scitex_dataset.general.zenodo import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {"hits": {"hits": [], "total": 0}}
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    datasets = fetch_all_datasets()

    assert len(datasets) == 0


# EOF
