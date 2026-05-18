#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/general/test_zenodo.py

"""Tests for Zenodo dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``httpx`` at the call site
(``zenodo.httpx.get``), so we swap ``zenodo.httpx`` at the module
namespace for the duration of each test.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import scitex_dataset.general.zenodo as _zenodo_mod
from scitex_dataset.general.zenodo import ZENODO_API, format_dataset

# ---------------------------------------------------------------------------
# Hand-rolled HTTP collaborator stubs (no unittest.mock)
# ---------------------------------------------------------------------------


class _StubResponse:
    """Stand-in for ``httpx.Response`` with just the surface zenodo uses."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StubHttpx:
    """Captures ``httpx.get`` calls and returns a queue of stub responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[tuple, dict]] = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self._responses:
            raise RuntimeError("StubHttpx ran out of queued responses")
        return self._responses.pop(0)


@contextmanager
def _swap_httpx(stub) -> Iterator[None]:
    saved = _zenodo_mod.httpx
    _zenodo_mod.httpx = stub  # type: ignore[assignment]
    try:
        yield
    finally:
        _zenodo_mod.httpx = saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zenodo_api_url_constant_matches_published_endpoint():
    # Arrange
    expected = "https://zenodo.org/api"
    # Act
    actual = ZENODO_API
    # Assert
    assert actual == expected


_FULL_RECORD = {
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
        {"size": 1_073_741_824},  # 1 GB
        {"size": 536_870_912},  # 0.5 GB
    ],
    "stats": {"views": 500, "downloads": 200},
    "links": {"html": "https://zenodo.org/record/12345"},
}


def test_format_dataset_full_record_returns_id_as_string():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["id"] == "12345"


def test_format_dataset_full_record_extracts_doi():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["doi"] == "10.5281/zenodo.12345"


def test_format_dataset_full_record_extracts_title_as_name():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["name"] == "Sample Neuroscience Dataset"


def test_format_dataset_full_record_extracts_description():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["description"] == "A dataset for testing."


def test_format_dataset_full_record_extracts_authors_list():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["authors"] == ["Smith, John", "Doe, Jane"]


def test_format_dataset_full_record_keywords_include_neuroscience():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert "neuroscience" in formatted["keywords"]


def test_format_dataset_full_record_license_id_passed_through():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["license"] == "cc-by-4.0"


def test_format_dataset_full_record_dataset_type_passed_through():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["dataset_type"] == "dataset"


def test_format_dataset_full_record_n_files_counts_two():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["n_files"] == 2


def test_format_dataset_full_record_size_bytes_sums_files():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["size_bytes"] == 1_073_741_824 + 536_870_912


def test_format_dataset_full_record_views_passed_through():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["views"] == 500


def test_format_dataset_full_record_downloads_passed_through():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["downloads"] == 200


def test_format_dataset_full_record_url_passed_through():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["url"] == "https://zenodo.org/record/12345"


def test_format_dataset_full_record_source_is_zenodo():
    # Arrange
    record = _FULL_RECORD
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["source"] == "zenodo"


def test_format_dataset_minimal_returns_id_as_string():
    # Arrange
    record = {"id": 99999}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["id"] == "99999"


def test_format_dataset_minimal_name_defaults_empty():
    # Arrange
    record = {"id": 99999}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["name"] == ""


def test_format_dataset_minimal_n_files_defaults_zero():
    # Arrange
    record = {"id": 99999}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["n_files"] == 0


def test_format_dataset_minimal_views_default_zero():
    # Arrange
    record = {"id": 99999}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["views"] == 0


def test_format_dataset_minimal_source_is_zenodo():
    # Arrange
    record = {"id": 99999}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["source"] == "zenodo"


def test_format_dataset_string_license_passed_through_unchanged():
    # Arrange
    record = {"id": 11111, "metadata": {"license": "MIT"}}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["license"] == "MIT"


def test_format_dataset_string_resource_type_passed_through_unchanged():
    # Arrange
    record = {"id": 22222, "metadata": {"resource_type": "dataset"}}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["dataset_type"] == "dataset"


def test_format_dataset_fallback_url_uses_record_id():
    # Arrange
    record = {"id": 33333}
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["url"] == "https://zenodo.org/record/33333"


def test_fetch_datasets_returns_hits_block_from_stub():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_datasets

    payload = {"hits": {"hits": [{"id": 1}, {"id": 2}], "total": 2}}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(query="neuroscience", page=1, size=25)
    # Assert
    assert "hits" in result


def test_fetch_datasets_returns_two_hits_when_stub_returns_two():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_datasets

    payload = {"hits": {"hits": [{"id": 1}, {"id": 2}], "total": 2}}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(query="neuroscience", page=1, size=25)
    # Assert
    assert len(result["hits"]["hits"]) == 2


def test_fetch_datasets_size_param_capped_at_25_for_anonymous():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_datasets

    stub = _StubHttpx([_StubResponse({"hits": {"hits": [], "total": 0}})])
    # Act
    with _swap_httpx(stub):
        fetch_datasets(size=100)
    call_url = stub.calls[0][0][0]
    # Assert
    assert "size=25" in call_url


def test_fetch_all_datasets_pagination_combines_two_pages_to_five():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_all_datasets

    page1 = {"hits": {"hits": [{"id": i} for i in range(1, 4)], "total": 5}}
    page2 = {"hits": {"hits": [{"id": i} for i in range(4, 6)], "total": 5}}
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 5


def test_fetch_all_datasets_pagination_calls_stub_twice():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_all_datasets

    page1 = {"hits": {"hits": [{"id": i} for i in range(1, 4)], "total": 5}}
    page2 = {"hits": {"hits": [{"id": i} for i in range(4, 6)], "total": 5}}
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        fetch_all_datasets()
    # Assert
    assert len(stub.calls) == 2


def test_fetch_all_datasets_respects_max_datasets_cap_of_three():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_all_datasets

    payload = {"hits": {"hits": [{"id": i} for i in range(1, 11)], "total": 100}}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets(max_datasets=3)
    # Assert
    assert len(datasets) == 3


def test_fetch_all_datasets_empty_response_returns_empty_list():
    # Arrange
    from scitex_dataset.general.zenodo import fetch_all_datasets

    stub = _StubHttpx([_StubResponse({"hits": {"hits": [], "total": 0}})])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 0


# EOF
