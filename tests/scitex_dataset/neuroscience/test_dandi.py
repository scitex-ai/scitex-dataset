#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/neuroscience/test_dandi.py

"""Tests for DANDI Archive dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``_httpx`` at the call site; we swap
``dandi._httpx`` at the module namespace via a real save/restore
context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import scitex_dataset.neuroscience.dandi as _dandi_mod
from scitex_dataset.neuroscience.dandi import DANDI_API, format_dataset


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
    saved = _dandi_mod._httpx
    _dandi_mod._httpx = stub  # type: ignore[assignment]
    try:
        yield
    finally:
        _dandi_mod._httpx = saved  # type: ignore[assignment]


def test_dandi_api_url_constant_matches_published_endpoint():
    # Arrange
    expected = "https://api.dandiarchive.org/api"
    # Act
    actual = DANDI_API
    # Assert
    assert actual == expected


def test_format_dataset_returns_identifier_as_id(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["id"] == "000001"


def test_format_dataset_returns_draft_name_as_name(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["name"] == "Sample Electrophysiology Data"


def test_format_dataset_returns_draft_version_as_version(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["version"] == "draft"


def test_format_dataset_returns_draft_status(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["status"] == "Valid"


def test_format_dataset_returns_contact_person_as_contact(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["contact"] == "researcher@example.com"


def test_format_dataset_returns_asset_count_as_n_assets(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["n_assets"] == 42


def test_format_dataset_computes_size_in_gigabytes(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["size_gb"] == 10.0


def test_format_dataset_url_contains_dandiset_identifier_path(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert "dandiarchive.org/dandiset/000001" in formatted["url"]


def test_format_dataset_passes_through_embargo_status(dandi_dandiset):
    # Arrange
    record = dandi_dandiset
    # Act
    formatted = format_dataset(record)
    # Assert
    assert formatted["embargo_status"] == "OPEN"


def test_format_dataset_missing_draft_falls_back_to_identifier_as_name():
    # Arrange
    dandiset = {
        "identifier": "000002",
        "created": "2021-01-01T00:00:00Z",
        "draft_version": None,
    }
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["name"] == "000002"


def test_format_dataset_missing_draft_returns_zero_assets():
    # Arrange
    dandiset = {
        "identifier": "000002",
        "created": "2021-01-01T00:00:00Z",
        "draft_version": None,
    }
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["n_assets"] == 0


def test_format_dataset_missing_draft_returns_zero_size_gb():
    # Arrange
    dandiset = {
        "identifier": "000002",
        "created": "2021-01-01T00:00:00Z",
        "draft_version": None,
    }
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["size_gb"] == 0.0


def test_format_dataset_minimal_returns_identifier_as_id_and_name():
    # Arrange
    dandiset = {"identifier": "000003"}
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["id"] == "000003"


def test_format_dataset_minimal_empty_contact_string():
    # Arrange
    dandiset = {"identifier": "000003"}
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["contact"] == ""


def test_format_dataset_minimal_embargo_status_is_none():
    # Arrange
    dandiset = {"identifier": "000003"}
    # Act
    formatted = format_dataset(dandiset)
    # Assert
    assert formatted["embargo_status"] is None


def test_fetch_datasets_returns_results_block_from_stub():
    # Arrange
    from scitex_dataset.neuroscience.dandi import fetch_datasets

    payload = {
        "results": [{"identifier": "000001"}, {"identifier": "000002"}],
        "next": None,
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(page=1, page_size=10)
    # Assert
    assert "results" in result


def test_fetch_datasets_returns_two_results_when_stub_returns_two():
    # Arrange
    from scitex_dataset.neuroscience.dandi import fetch_datasets

    payload = {
        "results": [{"identifier": "000001"}, {"identifier": "000002"}],
        "next": None,
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        result = fetch_datasets(page=1, page_size=10)
    # Assert
    assert len(result["results"]) == 2


def test_fetch_all_datasets_pagination_combines_two_pages_to_five():
    # Arrange
    from scitex_dataset.neuroscience.dandi import fetch_all_datasets

    page1 = {
        "results": [{"identifier": f"00000{i}"} for i in range(1, 4)],
        "next": "http://next-page",
    }
    page2 = {
        "results": [{"identifier": f"00000{i}"} for i in range(4, 6)],
        "next": None,
    }
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets()
    # Assert
    assert len(datasets) == 5


def test_fetch_all_datasets_pagination_calls_stub_twice():
    # Arrange
    from scitex_dataset.neuroscience.dandi import fetch_all_datasets

    page1 = {
        "results": [{"identifier": f"00000{i}"} for i in range(1, 4)],
        "next": "http://next-page",
    }
    page2 = {
        "results": [{"identifier": f"00000{i}"} for i in range(4, 6)],
        "next": None,
    }
    stub = _StubHttpx([_StubResponse(page1), _StubResponse(page2)])
    # Act
    with _swap_httpx(stub):
        fetch_all_datasets()
    # Assert
    assert len(stub.calls) == 2


def test_fetch_all_datasets_respects_max_datasets_cap_of_five():
    # Arrange
    from scitex_dataset.neuroscience.dandi import fetch_all_datasets

    payload = {
        "results": [{"identifier": f"00000{i}"} for i in range(1, 11)],
        "next": "http://next-page",
    }
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        datasets = fetch_all_datasets(max_datasets=5)
    # Assert
    assert len(datasets) == 5


# EOF
