#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_openneuro.py

"""Tests for OpenNeuro dataset fetcher."""

from unittest.mock import MagicMock, patch

from scitex_dataset import OPENNEURO_API, format_dataset
from scitex_dataset.neuroscience.openneuro import _make_query


def test_openneuro_api_url():
    """Test that API URL is correct."""
    assert OPENNEURO_API == "https://openneuro.org/crn/graphql"


def test_make_query_no_cursor():
    """Test GraphQL query generation without cursor."""
    query = _make_query(first=10)
    assert "first: 10" in query
    assert "after:" not in query


def test_make_query_with_cursor():
    """Test GraphQL query generation with cursor."""
    query = _make_query(first=5, after="abc123")
    assert "first: 5" in query
    assert 'after: "abc123"' in query


def test_fetch_datasets_mocked(mock_httpx_get):
    """Test fetch_datasets with mocked HTTP."""
    from scitex_dataset.neuroscience.openneuro import fetch_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": "ds000001"}}, {"node": {"id": "ds000002"}}],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    mock_response.raise_for_status = MagicMock()

    # OpenNeuro uses httpx.post, not httpx.get
    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = fetch_datasets(first=2)

    assert "data" in result
    assert len(result["data"]["datasets"]["edges"]) == 2
    mock_post.assert_called_once()


def test_fetch_all_datasets_pagination(mock_httpx_get):
    """Test fetch_all_datasets handles pagination."""
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    page1_response = MagicMock()
    page1_response.json.return_value = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds00000{i}"}} for i in range(1, 4)],
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
            }
        }
    }
    page1_response.raise_for_status = MagicMock()

    page2_response = MagicMock()
    page2_response.json.return_value = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds00000{i}"}} for i in range(4, 6)],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    page2_response.raise_for_status = MagicMock()

    with patch("httpx.post", side_effect=[page1_response, page2_response]):
        datasets = fetch_all_datasets(batch_size=3)

    assert len(datasets) == 5


def test_fetch_all_datasets_max_limit(mock_httpx_get):
    """Test fetch_all_datasets respects max_datasets."""
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds{i:06d}"}} for i in range(1, 11)],
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor"},
            }
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        datasets = fetch_all_datasets(max_datasets=5)

    assert len(datasets) >= 5


def test_fetch_all_datasets_graphql_error(mock_httpx_get):
    """Test fetch_all_datasets handles GraphQL errors."""
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    mock_response = MagicMock()
    mock_response.json.return_value = {"errors": [{"message": "Query too complex"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        datasets = fetch_all_datasets()

    assert len(datasets) == 0


def test_format_dataset():
    """Test formatting a dataset node."""
    node = {
        "id": "ds000001",
        "name": "Test Dataset",
        "created": "2020-01-01T00:00:00Z",
        "public": True,
        "publishDate": "2020-01-01T00:00:00Z",
        "analytics": {"views": 100, "downloads": 50},
        "draft": {
            "modified": "2020-06-01T00:00:00Z",
            "readme": "Test README content",
            "description": {
                "Name": "Test Dataset",
                "BIDSVersion": "1.6.0",
                "License": "CC0",
                "Authors": ["Author One", "Author Two"],
            },
            "summary": {
                "modalities": ["mri"],
                "primaryModality": "mri",
                "subjects": ["sub-01", "sub-02", "sub-03"],
                "tasks": ["rest"],
                "size": 1073741824,  # 1 GB
                "totalFiles": 100,
            },
        },
    }

    formatted = format_dataset(node)

    assert formatted["id"] == "ds000001"
    assert formatted["name"] == "Test Dataset"
    assert formatted["n_subjects"] == 3
    assert formatted["size_gb"] == 1.0
    assert formatted["views"] == 100
    assert formatted["downloads"] == 50


def test_format_dataset_missing_fields():
    """Test formatting with missing optional fields."""
    node = {
        "id": "ds000002",
        "name": None,
        "draft": None,
    }

    formatted = format_dataset(node)

    assert formatted["id"] == "ds000002"
    assert formatted["name"] == "N/A"
    assert formatted["n_subjects"] == 0
    assert formatted["size_gb"] == 0.0


# EOF
