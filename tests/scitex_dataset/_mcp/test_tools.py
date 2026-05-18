#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/_mcp/test_tools.py

"""Tests for MCP tools registration.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. Each
collaborator is swapped at the module namespace via a real save/restore
context manager, mirroring the pattern used in
``scitex-agent-container/tests/.../test__send.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import scitex_dataset.database as _db_mod
import scitex_dataset.neuroscience.dandi as _dandi_mod
import scitex_dataset.neuroscience.openneuro as _on_mod
import scitex_dataset.neuroscience.physionet as _pn_mod

# ---------------------------------------------------------------------------
# Hand-rolled MCP stub (not a unittest.mock object)
# ---------------------------------------------------------------------------


class _MockMCP:
    """FastMCP-shaped tool collector used by registration assertions."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        """Decorator that captures tool functions."""

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


# ---------------------------------------------------------------------------
# Collaborator swaps (test seams — no mocks, no monkeypatch)
# ---------------------------------------------------------------------------


@contextmanager
def _swap_openneuro(fetch_all, format_one) -> Iterator[None]:
    saved_fetch = _on_mod.fetch_all_datasets
    saved_format = _on_mod.format_dataset
    _on_mod.fetch_all_datasets = fetch_all  # type: ignore[assignment]
    _on_mod.format_dataset = format_one  # type: ignore[assignment]
    try:
        yield
    finally:
        _on_mod.fetch_all_datasets = saved_fetch  # type: ignore[assignment]
        _on_mod.format_dataset = saved_format  # type: ignore[assignment]


@contextmanager
def _swap_dandi(fetch_all, format_one) -> Iterator[None]:
    saved_fetch = _dandi_mod.fetch_all_datasets
    saved_format = _dandi_mod.format_dataset
    _dandi_mod.fetch_all_datasets = fetch_all  # type: ignore[assignment]
    _dandi_mod.format_dataset = format_one  # type: ignore[assignment]
    try:
        yield
    finally:
        _dandi_mod.fetch_all_datasets = saved_fetch  # type: ignore[assignment]
        _dandi_mod.format_dataset = saved_format  # type: ignore[assignment]


@contextmanager
def _swap_physionet(fetch_all, format_one) -> Iterator[None]:
    saved_fetch = _pn_mod.fetch_all_datasets
    saved_format = _pn_mod.format_dataset
    _pn_mod.fetch_all_datasets = fetch_all  # type: ignore[assignment]
    _pn_mod.format_dataset = format_one  # type: ignore[assignment]
    try:
        yield
    finally:
        _pn_mod.fetch_all_datasets = saved_fetch  # type: ignore[assignment]
        _pn_mod.format_dataset = saved_format  # type: ignore[assignment]


@contextmanager
def _swap_db_get_stats(fn) -> Iterator[None]:
    saved = _db_mod.get_stats
    _db_mod.get_stats = fn  # type: ignore[assignment]
    try:
        yield
    finally:
        _db_mod.get_stats = saved  # type: ignore[assignment]


@contextmanager
def _swap_db_search(fn) -> Iterator[None]:
    saved = _db_mod.search
    _db_mod.search = fn  # type: ignore[assignment]
    try:
        yield
    finally:
        _db_mod.search = saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _register_all() -> _MockMCP:
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    register_all_tools(m)
    return m


def test_register_all_tools_includes_openneuro_fetch():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "openneuro_fetch" in m.tools


def test_register_all_tools_includes_dandi_fetch():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "dandi_fetch" in m.tools


def test_register_all_tools_includes_physionet_fetch():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "physionet_fetch" in m.tools


def test_register_all_tools_includes_filter_results():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "filter_results" in m.tools


def test_register_all_tools_includes_list_sources():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "list_sources" in m.tools


def test_register_all_tools_includes_db_build():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "db_build" in m.tools


def test_register_all_tools_includes_db_search():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "db_search" in m.tools


def test_register_all_tools_includes_db_show_stats():
    # Arrange
    from scitex_dataset._mcp.tools import register_all_tools

    m = _MockMCP()
    # Act
    register_all_tools(m)
    # Assert
    assert "db_show_stats" in m.tools


def test_list_sources_payload_contains_all_sources():
    # Arrange
    from scitex_dataset._sources import ALL_SOURCES

    m = _register_all()
    # Act
    result = m.tools["list_sources"]()
    # Assert
    assert all(src in result["sources"] for src in ALL_SOURCES)


def test_list_sources_count_equals_all_sources_length():
    # Arrange
    from scitex_dataset._sources import ALL_SOURCES

    m = _register_all()
    # Act
    result = m.tools["list_sources"]()
    # Assert
    assert result["count"] == len(ALL_SOURCES)


def test_openneuro_fetch_returns_swapped_length():
    # Arrange
    calls: list[dict] = []

    def fake_fetch(**kw):
        calls.append(kw)
        return [{"id": "ds001"}, {"id": "ds002"}]

    def fake_format(x):
        return {"id": x["id"], "name": f"Dataset {x['id']}"}

    m = _register_all()
    # Act
    with _swap_openneuro(fake_fetch, fake_format):
        result = m.tools["openneuro_fetch"](max_datasets=2)
    # Assert
    assert len(result) == 2


def test_openneuro_fetch_preserves_first_id_from_swapped_fetch():
    # Arrange
    def fake_fetch(**kw):
        del kw
        return [{"id": "ds001"}, {"id": "ds002"}]

    def fake_format(x):
        return {"id": x["id"], "name": f"Dataset {x['id']}"}

    m = _register_all()
    # Act
    with _swap_openneuro(fake_fetch, fake_format):
        result = m.tools["openneuro_fetch"](max_datasets=2)
    # Assert
    assert result[0]["id"] == "ds001"


def test_dandi_fetch_returns_one_item_when_swapped_fetch_returns_one():
    # Arrange
    def fake_fetch(**kw):
        del kw
        return [{"identifier": "000001"}]

    def fake_format(x):
        del x
        return {"id": "000001", "name": "DANDI Dataset"}

    m = _register_all()
    # Act
    with _swap_dandi(fake_fetch, fake_format):
        result = m.tools["dandi_fetch"](max_datasets=1)
    # Assert
    assert len(result) == 1


def test_dandi_fetch_returns_id_from_swapped_format():
    # Arrange
    def fake_fetch(**kw):
        del kw
        return [{"identifier": "000001"}]

    def fake_format(x):
        del x
        return {"id": "000001", "name": "DANDI Dataset"}

    m = _register_all()
    # Act
    with _swap_dandi(fake_fetch, fake_format):
        result = m.tools["dandi_fetch"](max_datasets=1)
    # Assert
    assert result[0]["id"] == "000001"


def test_physionet_fetch_returns_one_item_when_swapped_fetch_returns_one():
    # Arrange
    def fake_fetch(**kw):
        del kw
        return [{"slug": "test-db"}]

    def fake_format(x):
        del x
        return {"id": "test-db", "name": "Test DB"}

    m = _register_all()
    # Act
    with _swap_physionet(fake_fetch, fake_format):
        result = m.tools["physionet_fetch"](max_datasets=1)
    # Assert
    assert len(result) == 1


def test_physionet_fetch_returns_id_from_swapped_format():
    # Arrange
    def fake_fetch(**kw):
        del kw
        return [{"slug": "test-db"}]

    def fake_format(x):
        del x
        return {"id": "test-db", "name": "Test DB"}

    m = _register_all()
    # Act
    with _swap_physionet(fake_fetch, fake_format):
        result = m.tools["physionet_fetch"](max_datasets=1)
    # Assert
    assert result[0]["id"] == "test-db"


def test_filter_results_by_modality_returns_only_eeg_entries(sample_datasets):
    # Arrange
    m = _register_all()
    # Act
    result = m.tools["filter_results"](
        datasets=sample_datasets,
        modality="eeg",
        limit=10,
    )
    # Assert
    assert all("eeg" in ds.get("modalities", []) for ds in result)


def test_filter_results_respects_min_subjects_floor(sample_datasets):
    # Arrange
    m = _register_all()
    # Act
    result = m.tools["filter_results"](
        datasets=sample_datasets,
        min_subjects=30,
        min_downloads=100,
        limit=10,
    )
    # Assert
    assert all(ds["n_subjects"] >= 30 for ds in result)


def test_filter_results_respects_min_downloads_floor(sample_datasets):
    # Arrange
    m = _register_all()
    # Act
    result = m.tools["filter_results"](
        datasets=sample_datasets,
        min_subjects=30,
        min_downloads=100,
        limit=10,
    )
    # Assert
    assert all(ds["downloads"] >= 100 for ds in result)


def test_db_show_stats_returns_exists_true_from_swapped_stats():
    # Arrange
    def fake_stats(**kw):
        del kw
        return {
            "exists": True,
            "total_datasets": 1_000,
            "by_source": {"openneuro": 600, "dandi": 300, "physionet": 100},
        }

    m = _register_all()
    # Act
    with _swap_db_get_stats(fake_stats):
        result = m.tools["db_show_stats"]()
    # Assert
    assert result["exists"] is True


def test_db_show_stats_returns_total_from_swapped_stats():
    # Arrange
    def fake_stats(**kw):
        del kw
        return {
            "exists": True,
            "total_datasets": 1_000,
            "by_source": {"openneuro": 600, "dandi": 300, "physionet": 100},
        }

    m = _register_all()
    # Act
    with _swap_db_get_stats(fake_stats):
        result = m.tools["db_show_stats"]()
    # Assert
    assert result["total_datasets"] == 1_000


def test_db_search_returns_one_item_from_swapped_search():
    # Arrange
    def fake_search(**kw):
        del kw
        return [{"id": "ds001", "name": "Test", "n_subjects": 25}]

    m = _register_all()
    # Act
    with _swap_db_search(fake_search):
        result = m.tools["db_search"](
            query="memory",
            modality="eeg",
            limit=10,
        )
    # Assert
    assert len(result) == 1


# EOF
