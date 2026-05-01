#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 23:58:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_dandi.py

"""Tests for DANDI Archive dataset fetcher."""

from unittest.mock import MagicMock

from scitex_dataset.neuroscience.dandi import DANDI_API, format_dataset


def test_dandi_api_url():
    """Test that API URL is correct."""
    assert DANDI_API == "https://api.dandiarchive.org/api"


def test_format_dataset(dandi_dandiset):
    """Test formatting a DANDI dandiset."""
    formatted = format_dataset(dandi_dandiset)

    assert formatted["id"] == "000001"
    assert formatted["name"] == "Sample Electrophysiology Data"
    assert formatted["version"] == "draft"
    assert formatted["status"] == "Valid"
    assert formatted["contact"] == "researcher@example.com"
    assert formatted["n_assets"] == 42
    assert formatted["size_gb"] == 10.0
    assert "dandiarchive.org/dandiset/000001" in formatted["url"]
    assert formatted["embargo_status"] == "OPEN"


def test_format_dataset_missing_draft():
    """Test formatting with missing draft version."""
    dandiset = {
        "identifier": "000002",
        "created": "2021-01-01T00:00:00Z",
        "draft_version": None,
    }

    formatted = format_dataset(dandiset)

    assert formatted["id"] == "000002"
    assert formatted["name"] == "000002"  # Falls back to identifier
    assert formatted["n_assets"] == 0
    assert formatted["size_gb"] == 0.0


def test_format_dataset_minimal():
    """Test formatting with minimal fields."""
    dandiset = {"identifier": "000003"}

    formatted = format_dataset(dandiset)

    assert formatted["id"] == "000003"
    assert formatted["name"] == "000003"
    assert formatted["contact"] == ""
    assert formatted["embargo_status"] is None


def test_fetch_datasets_mocked(mock_httpx_get):
    """Test fetch_datasets with mocked HTTP."""
    from scitex_dataset.neuroscience.dandi import fetch_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"identifier": "000001"}, {"identifier": "000002"}],
        "next": None,
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    result = fetch_datasets(page=1, page_size=10)

    assert "results" in result
    assert len(result["results"]) == 2
    mock_httpx_get.assert_called_once()


def test_fetch_all_datasets_pagination(mock_httpx_get):
    """Test fetch_all_datasets handles pagination."""
    from scitex_dataset.neuroscience.dandi import fetch_all_datasets

    # First page has next, second page doesn't
    page1_response = MagicMock()
    page1_response.json.return_value = {
        "results": [{"identifier": f"00000{i}"} for i in range(1, 4)],
        "next": "http://next-page",
    }
    page1_response.raise_for_status = MagicMock()

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "results": [{"identifier": f"00000{i}"} for i in range(4, 6)],
        "next": None,
    }
    page2_response.raise_for_status = MagicMock()

    mock_httpx_get.side_effect = [page1_response, page2_response]

    datasets = fetch_all_datasets()

    assert len(datasets) == 5
    assert mock_httpx_get.call_count == 2


def test_fetch_all_datasets_max_limit(mock_httpx_get):
    """Test fetch_all_datasets respects max_datasets."""
    from scitex_dataset.neuroscience.dandi import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"identifier": f"00000{i}"} for i in range(1, 11)],
        "next": "http://next-page",
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_get.return_value = mock_response

    datasets = fetch_all_datasets(max_datasets=5)

    assert len(datasets) == 5


# EOF
