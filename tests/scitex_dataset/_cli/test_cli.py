#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 23:58:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_cli.py

"""Tests for CLI commands."""

from unittest.mock import patch

from scitex_dataset._cli import main


def test_version(cli_runner):
    """Test --version flag."""
    result = cli_runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "scitex-dataset" in result.output


def test_help(cli_runner):
    """Test --help flag."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "OpenNeuro" in result.output or "openneuro" in result.output
    assert "DANDI" in result.output or "dandi" in result.output


def test_help_recursive(cli_runner):
    """Test --help-recursive flag."""
    result = cli_runner.invoke(main, ["--help-recursive"])
    assert result.exit_code == 0
    # Should contain all subcommands
    assert "openneuro" in result.output
    assert "dandi" in result.output
    assert "physionet" in result.output
    assert "db" in result.output
    assert "mcp" in result.output


def test_no_args_shows_help(cli_runner):
    """Test that no args shows help."""
    result = cli_runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_openneuro_help(cli_runner):
    """Test fetch-openneuro --help."""
    result = cli_runner.invoke(main, ["fetch-openneuro", "--help"])
    assert result.exit_code == 0
    assert "max-datasets" in result.output
    assert "output" in result.output


@patch("scitex_dataset.neuroscience.openneuro.fetch_all_datasets")
@patch("scitex_dataset.neuroscience.openneuro.format_dataset")
def test_openneuro_fetch(mock_format, mock_fetch, cli_runner, tmp_path):
    """Test fetch-openneuro command fetches datasets."""
    mock_fetch.return_value = [{"id": "ds001"}, {"id": "ds002"}]
    mock_format.side_effect = lambda x: {"id": x["id"], "name": f"Dataset {x['id']}"}

    result = cli_runner.invoke(main, ["fetch-openneuro", "-n", "2"])

    assert result.exit_code == 0
    assert "2 datasets" in result.output
    mock_fetch.assert_called_once()


@patch("scitex_dataset.neuroscience.openneuro.fetch_all_datasets")
@patch("scitex_dataset.neuroscience.openneuro.format_dataset")
def test_openneuro_output_file(mock_format, mock_fetch, cli_runner, tmp_path):
    """Test fetch-openneuro command writes to file."""
    output_file = tmp_path / "datasets.json"
    mock_fetch.return_value = [{"id": "ds001"}]
    mock_format.return_value = {"id": "ds001", "name": "Test"}

    result = cli_runner.invoke(
        main, ["fetch-openneuro", "-n", "1", "-o", str(output_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Saved" in result.output


def test_dandi_help(cli_runner):
    """Test fetch-dandi --help."""
    result = cli_runner.invoke(main, ["fetch-dandi", "--help"])
    assert result.exit_code == 0
    assert "NWB" in result.output or "DANDI" in result.output


def test_physionet_help(cli_runner):
    """Test fetch-physionet --help."""
    result = cli_runner.invoke(main, ["fetch-physionet", "--help"])
    assert result.exit_code == 0
    assert "PhysioNet" in result.output or "physionet" in result.output


def test_db_help(cli_runner):
    """Test db --help."""
    result = cli_runner.invoke(main, ["db", "--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "search" in result.output
    assert "show-stats" in result.output


def test_db_help_recursive(cli_runner):
    """Test db --help-recursive."""
    result = cli_runner.invoke(main, ["db", "--help-recursive"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "search" in result.output


def test_db_stats_no_db(cli_runner, tmp_path, monkeypatch):
    """Test db show-stats when no database exists."""
    # Point to non-existent database
    monkeypatch.setattr(
        "scitex_dataset.database.DEFAULT_DB_PATH",
        tmp_path / "nonexistent.db",
    )

    result = cli_runner.invoke(main, ["db", "show-stats"])

    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "not built" in result.output.lower()


@patch("scitex_dataset.database.search")
def test_db_search(mock_search, cli_runner):
    """Test db search command."""
    mock_search.return_value = [
        {"id": "ds001", "name": "Test Dataset", "n_subjects": 25, "downloads": 100}
    ]

    result = cli_runner.invoke(main, ["db", "search", "memory"])

    assert result.exit_code == 0
    mock_search.assert_called_once()


@patch("scitex_dataset.database.search")
def test_db_search_no_results(mock_search, cli_runner):
    """Test db search with no results."""
    mock_search.return_value = []

    result = cli_runner.invoke(main, ["db", "search", "nonexistent"])

    assert result.exit_code == 0
    assert "No datasets found" in result.output


def test_mcp_help(cli_runner):
    """Test mcp --help."""
    result = cli_runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "list-tools" in result.output
    assert "doctor" in result.output


def test_mcp_install(cli_runner):
    """Test mcp install command."""
    result = cli_runner.invoke(main, ["mcp", "install"])
    assert result.exit_code == 0
    assert "Installation" in result.output or "install" in result.output.lower()


def test_mcp_install_claude_code(cli_runner):
    """Test mcp install --claude-code."""
    result = cli_runner.invoke(main, ["mcp", "install", "--claude-code"])
    assert result.exit_code == 0
    assert "scitex-dataset" in result.output
    assert "mcp" in result.output


def test_mcp_doctor_no_fastmcp(cli_runner, monkeypatch):
    """Test mcp doctor when fastmcp not installed."""
    import sys

    # Temporarily remove fastmcp from modules
    original = sys.modules.get("fastmcp")
    sys.modules["fastmcp"] = None

    result = cli_runner.invoke(main, ["mcp", "doctor"])

    # Restore
    if original:
        sys.modules["fastmcp"] = original
    else:
        sys.modules.pop("fastmcp", None)

    # Should report fastmcp not installed
    assert (
        "fastmcp" in result.output.lower() or "not installed" in result.output.lower()
    )


def test_completion_help(cli_runner):
    """Test print-tab-completion --help."""
    result = cli_runner.invoke(main, ["print-tab-completion", "--help"])
    assert result.exit_code == 0
    assert "bash" in result.output
    assert "zsh" in result.output


def test_completion_bash(cli_runner):
    """Test print-tab-completion --shell bash generates script."""
    result = cli_runner.invoke(main, ["print-tab-completion", "--shell", "bash"])
    # Should produce some output (completion script or error)
    assert result.exit_code == 0


def test_list_python_apis(cli_runner):
    """Test list-python-apis command."""
    result = cli_runner.invoke(main, ["list-python-apis"])
    assert result.exit_code == 0
    assert "scitex_dataset" in result.output
    assert "[M]" in result.output  # Module marker
    assert "[F]" in result.output  # Function marker


def test_list_python_apis_verbose(cli_runner):
    """Test list-python-apis with verbose output."""
    result = cli_runner.invoke(main, ["list-python-apis", "-v"])
    assert result.exit_code == 0
    assert "fetch_all_datasets" in result.output


def test_list_python_apis_json(cli_runner):
    """Test list-python-apis JSON output."""
    result = cli_runner.invoke(main, ["list-python-apis", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Type" in data[0]


# EOF
