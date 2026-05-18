#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/neuroscience/test_physionet.py

"""Tests for PhysioNet dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``_httpx`` at the call site; we swap
``physionet._httpx`` at the module namespace via a real save/restore
context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import scitex_dataset.neuroscience.physionet as _pn_mod
from scitex_dataset.neuroscience.physionet import PHYSIONET_API, format_dataset


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _StubHttpx:
    """Captures ``_httpx.get`` calls and returns a queue of stub responses."""

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
    saved = _pn_mod._httpx
    _pn_mod._httpx = stub  # type: ignore[assignment]
    try:
        yield
    finally:
        _pn_mod._httpx = saved  # type: ignore[assignment]


def test_physionet_api_url_constant_matches_published_endpoint():
    # Arrange
    expected = "https://physionet.org"
    # Act
    actual = PHYSIONET_API
    # Assert
    assert actual == expected


def test_format_dataset_returns_slug_as_id(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["id"] == "sample-eeg-db"


def test_format_dataset_returns_title_as_name(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["name"] == "Sample EEG Database"


def test_format_dataset_passes_through_version(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["version"] == "1.0.0"


def test_format_dataset_lowercased_abstract_mentions_epilepsy(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert "epilepsy" in formatted["abstract"].lower()


def test_format_dataset_passes_through_doi(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["doi"] == "10.13026/xxxx-yyyy"


def test_format_dataset_license_string_contains_attribution(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert "Attribution" in formatted["license"]


def test_format_dataset_returns_subject_count_as_n_subjects(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["n_subjects"] == 100


def test_format_dataset_returns_record_count_as_n_records(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["n_records"] == 500


def test_format_dataset_computes_size_in_gigabytes(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["size_gb"] == 20.0


def test_format_dataset_url_contains_content_slug_path(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert "physionet.org/content/sample-eeg-db" in formatted["url"]


def test_format_dataset_data_access_passed_through(physionet_database):
    # Arrange
    record = physionet_database
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["data_access"] == "open"


def test_format_dataset_string_license_passes_through_unchanged():
    # Arrange
    database = {
        "slug": "test-db",
        "title": "Test Database",
        "license": "ODC-By v1.0",
    }
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["license"] == "ODC-By v1.0"


def test_format_dataset_minimal_id_defaults_empty_string():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["id"] == ""


def test_format_dataset_minimal_name_defaults_to_na_marker():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["name"] == "N/A"


def test_format_dataset_minimal_n_subjects_defaults_zero():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["n_subjects"] == 0


def test_format_dataset_minimal_n_records_defaults_zero():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["n_records"] == 0


def test_format_dataset_minimal_size_gb_defaults_zero_point_zero():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["size_gb"] == 0.0


def test_format_dataset_minimal_url_defaults_empty_string():
    # Arrange
    database = {}
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["url"] == ""


def test_format_dataset_alternate_key_short_name_used_as_id():
    # Arrange
    database = {
        "short_name": "alt-db",
        "name": "Alternate Database",
        "description": "An alternate description.",
    }
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["id"] == "alt-db"


def test_format_dataset_alternate_key_name_used_when_title_missing():
    # Arrange
    database = {
        "short_name": "alt-db",
        "name": "Alternate Database",
        "description": "An alternate description.",
    }
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["name"] == "Alternate Database"


def test_format_dataset_alternate_key_description_used_as_abstract():
    # Arrange
    database = {
        "short_name": "alt-db",
        "name": "Alternate Database",
        "description": "An alternate description.",
    }
    # Act
    formatted = format_dataset(database)
    # Assert
    assert formatted["abstract"] == "An alternate description."


def test_fetch_datasets_returns_list_when_stub_returns_list():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_datasets

    payload = [
        {"slug": "db1", "title": "Database 1"},
        {"slug": "db2", "title": "Database 2"},
    ]
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(page=1)
    # Assert
    assert isinstance(result, list)


def test_fetch_datasets_returns_two_items_when_stub_returns_two():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_datasets

    payload = [
        {"slug": "db1", "title": "Database 1"},
        {"slug": "db2", "title": "Database 2"},
    ]
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(page=1)
    # Assert
    assert len(result) == 2


def test_fetch_all_datasets_list_response_returns_five_items():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    payload = [{"slug": f"db{i}"} for i in range(1, 6)]
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 5


def test_fetch_all_datasets_list_response_calls_stub_once():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    payload = [{"slug": f"db{i}"} for i in range(1, 6)]
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        fetch_all_datasets()
    # Assert
    assert len(stub.calls) == 1


def test_fetch_all_datasets_respects_max_datasets_cap_of_three():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    payload = [{"slug": f"db{i}"} for i in range(1, 11)]
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets(max_datasets=3)
    # Assert
    assert len(datasets) == 3


def test_fetch_all_datasets_paginated_dict_response_combines_two_pages_to_five():
    # Arrange
    from scitex_dataset.neuroscience.physionet import fetch_all_datasets

    page1 = {
        "results": [{"slug": f"db{i}"} for i in range(1, 4)],
        "next": "http://next",
    }
    page2 = {"results": [{"slug": f"db{i}"} for i in range(4, 6)], "next": None}
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 5


# EOF
