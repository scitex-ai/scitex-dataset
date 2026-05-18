#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/neuroscience/test_openneuro.py

"""Tests for OpenNeuro dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``_httpx`` at the call site
(``_httpx.post``); we swap ``openneuro._httpx`` at the module
namespace via a real save/restore context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import scitex_dataset.neuroscience.openneuro as _on_mod
from scitex_dataset import OPENNEURO_API, format_dataset
from scitex_dataset.neuroscience.openneuro import _make_query


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StubHttpx:
    """Captures ``_httpx.post`` calls and returns a queue of stub responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[tuple, dict]] = []
        # Pass through the real exception classes used by control flow.
        import httpx as _real

        self.HTTPStatusError = _real.HTTPStatusError
        self.RequestError = _real.RequestError

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self._responses:
            raise RuntimeError("StubHttpx ran out of queued responses")
        return self._responses.pop(0)


@contextmanager
def _swap_httpx(stub) -> Iterator[None]:
    saved = _on_mod._httpx
    _on_mod._httpx = stub  # type: ignore[assignment]
    try:
        yield
    finally:
        _on_mod._httpx = saved  # type: ignore[assignment]


def test_openneuro_api_url_constant_matches_published_endpoint():
    # Arrange
    expected = "https://openneuro.org/crn/graphql"
    # Act
    actual = OPENNEURO_API
    # Assert
    assert actual == expected


def test_make_query_without_cursor_embeds_first_argument():
    # Arrange
    expected = "first: 10"
    # Act
    query = _make_query(first=10)
    # Assert
    assert expected in query


def test_make_query_without_cursor_omits_after_argument():
    # Arrange
    forbidden = "after:"
    # Act
    query = _make_query(first=10)
    # Assert
    assert forbidden not in query


def test_make_query_with_cursor_embeds_first_argument():
    # Arrange
    expected = "first: 5"
    # Act
    query = _make_query(first=5, after="abc123")
    # Assert
    assert expected in query


def test_make_query_with_cursor_embeds_after_argument():
    # Arrange
    expected = 'after: "abc123"'
    # Act
    query = _make_query(first=5, after="abc123")
    # Assert
    assert expected in query


def test_fetch_datasets_returns_data_block_from_stub():
    # Arrange
    from scitex_dataset.neuroscience.openneuro import fetch_datasets

    payload = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": "ds000001"}}, {"node": {"id": "ds000002"}}],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(first=2)
    # Assert
    assert "data" in result


def test_fetch_datasets_returns_two_edges_when_stub_returns_two():
    # Arrange
    from scitex_dataset.neuroscience.openneuro import fetch_datasets

    payload = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": "ds000001"}}, {"node": {"id": "ds000002"}}],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(first=2)
    # Assert
    assert len(result["data"]["datasets"]["edges"]) == 2


def test_fetch_all_datasets_pagination_combines_two_pages_to_five():
    # Arrange
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    page1 = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds00000{i}"}} for i in range(1, 4)],
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
            }
        }
    }
    page2 = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds00000{i}"}} for i in range(4, 6)],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets(batch_size=3)
    # Assert
    assert len(datasets) == 5


def test_fetch_all_datasets_respects_max_datasets_cap_of_five():
    # Arrange
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    payload = {
        "data": {
            "datasets": {
                "edges": [{"node": {"id": f"ds{i:06d}"}} for i in range(1, 11)],
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor"},
            }
        }
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets(max_datasets=5)
    # Assert
    assert len(datasets) >= 5


def test_fetch_all_datasets_graphql_error_returns_empty_list():
    # Arrange
    from scitex_dataset.neuroscience.openneuro import fetch_all_datasets

    payload = {"errors": [{"message": "Query too complex"}]}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 0


_NODE_FULL = {
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
            "size": 1_073_741_824,  # 1 GB
            "totalFiles": 100,
        },
    },
}


def test_format_dataset_returns_node_id_unchanged():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["id"] == "ds000001"


def test_format_dataset_returns_node_name_unchanged():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["name"] == "Test Dataset"


def test_format_dataset_counts_subjects_list():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["n_subjects"] == 3


def test_format_dataset_computes_size_in_gigabytes_to_one_decimal():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["size_gb"] == 1.0


def test_format_dataset_passes_through_view_count():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["views"] == 100


def test_format_dataset_passes_through_download_count():
    # Arrange
    node = _NODE_FULL
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["downloads"] == 50


def test_format_dataset_missing_fields_id_passed_through():
    # Arrange
    node = {"id": "ds000002", "name": None, "draft": None}
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["id"] == "ds000002"


def test_format_dataset_missing_fields_name_defaults_to_na_marker():
    # Arrange
    node = {"id": "ds000002", "name": None, "draft": None}
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["name"] == "N/A"


def test_format_dataset_missing_fields_n_subjects_defaults_zero():
    # Arrange
    node = {"id": "ds000002", "name": None, "draft": None}
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["n_subjects"] == 0


def test_format_dataset_missing_fields_size_gb_defaults_zero_point_zero():
    # Arrange
    node = {"id": "ds000002", "name": None, "draft": None}
    # Act
    formatted = format_dataset(node)
    # Assert
    assert formatted["size_gb"] == 0.0


# EOF
