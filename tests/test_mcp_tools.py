#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 23:58:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_mcp_tools.py

"""Tests for MCP tools registration."""

from unittest.mock import patch


class MockMCP:
    """Mock FastMCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        """Decorator that captures tool functions."""

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


def test_register_all_tools():
    """Test that all tools are registered."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    # Check expected tools are registered
    expected_tools = [
        "dataset_openneuro_fetch",
        "dataset_dandi_fetch",
        "dataset_physionet_fetch",
        "dataset_search",
        "dataset_list_sources",
        "dataset_db_build",
        "dataset_db_search",
        "dataset_db_stats",
    ]

    for tool_name in expected_tools:
        assert tool_name in mock_mcp.tools, f"Missing tool: {tool_name}"


def test_dataset_list_sources():
    """Test dataset_list_sources tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_list_sources"]()

    assert "sources" in result
    assert "openneuro" in result["sources"]
    assert "dandi" in result["sources"]
    assert "physionet" in result["sources"]
    assert "zenodo" in result["sources"]
    assert "geo" in result["sources"]
    assert "chembl" in result["sources"]
    assert "clinicaltrials" in result["sources"]
    assert result["count"] == 7


@patch("scitex_dataset.neuroscience.openneuro.fetch_all_datasets")
@patch("scitex_dataset.neuroscience.openneuro.format_dataset")
def test_dataset_openneuro_fetch(mock_format, mock_fetch):
    """Test dataset_openneuro_fetch tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_fetch.return_value = [{"id": "ds001"}, {"id": "ds002"}]
    mock_format.side_effect = lambda x: {"id": x["id"], "name": f"Dataset {x['id']}"}

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_openneuro_fetch"](max_datasets=2)

    assert len(result) == 2
    assert result[0]["id"] == "ds001"
    mock_fetch.assert_called_once()


@patch("scitex_dataset.neuroscience.dandi.fetch_all_datasets")
@patch("scitex_dataset.neuroscience.dandi.format_dataset")
def test_dataset_dandi_fetch(mock_format, mock_fetch):
    """Test dataset_dandi_fetch tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_fetch.return_value = [{"identifier": "000001"}]
    mock_format.return_value = {"id": "000001", "name": "DANDI Dataset"}

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_dandi_fetch"](max_datasets=1)

    assert len(result) == 1
    assert result[0]["id"] == "000001"


@patch("scitex_dataset.neuroscience.physionet.fetch_all_datasets")
@patch("scitex_dataset.neuroscience.physionet.format_dataset")
def test_dataset_physionet_fetch(mock_format, mock_fetch):
    """Test dataset_physionet_fetch tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_fetch.return_value = [{"slug": "test-db"}]
    mock_format.return_value = {"id": "test-db", "name": "Test DB"}

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_physionet_fetch"](max_datasets=1)

    assert len(result) == 1
    assert result[0]["id"] == "test-db"


def test_dataset_search(sample_datasets):
    """Test dataset_search tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    # Search by modality
    result = mock_mcp.tools["dataset_search"](
        datasets=sample_datasets,
        modality="eeg",
        limit=10,
    )

    assert len(result) > 0
    assert all("eeg" in ds.get("modalities", []) for ds in result)


def test_dataset_search_with_filters(sample_datasets):
    """Test dataset_search with multiple filters."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_search"](
        datasets=sample_datasets,
        min_subjects=30,
        min_downloads=100,
        limit=10,
    )

    for ds in result:
        assert ds["n_subjects"] >= 30
        assert ds["downloads"] >= 100


@patch("scitex_dataset.database.get_stats")
def test_dataset_db_stats(mock_stats):
    """Test dataset_db_stats tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_stats.return_value = {
        "exists": True,
        "total_datasets": 1000,
        "by_source": {"openneuro": 600, "dandi": 300, "physionet": 100},
    }

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_db_stats"]()

    assert result["exists"] is True
    assert result["total_datasets"] == 1000


@patch("scitex_dataset.database.search")
def test_dataset_db_search(mock_search):
    """Test dataset_db_search tool."""
    from scitex_dataset._mcp.tools import register_all_tools

    mock_search.return_value = [{"id": "ds001", "name": "Test", "n_subjects": 25}]

    mock_mcp = MockMCP()
    register_all_tools(mock_mcp)

    result = mock_mcp.tools["dataset_db_search"](
        query="memory",
        modality="eeg",
        limit=10,
    )

    assert len(result) == 1
    mock_search.assert_called_once()


# EOF
