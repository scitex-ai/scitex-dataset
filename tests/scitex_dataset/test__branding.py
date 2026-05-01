#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-14 09:15:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_branding.py

"""Tests for branding module."""

import os
from unittest.mock import patch

from scitex_dataset._branding import (
    ENV_PREFIX,
    MCP_SERVER_NAME,
    PACKAGE_NAME,
    get_env,
    get_mcp_instructions,
    get_mcp_server_name,
)


def test_package_name():
    """Test package name constant."""
    assert PACKAGE_NAME == "scitex-dataset"


def test_env_prefix():
    """Test environment variable prefix."""
    assert ENV_PREFIX == "SCITEX_DATASET"


def test_mcp_server_name_constant():
    """Test MCP server name constant."""
    assert MCP_SERVER_NAME == "scitex-dataset"


def test_get_env_with_value():
    """Test get_env retrieves environment variable."""
    with patch.dict(os.environ, {"SCITEX_DATASET_TEST_KEY": "test_value"}):
        result = get_env("TEST_KEY")
    assert result == "test_value"


def test_get_env_default():
    """Test get_env returns default when not set."""
    result = get_env("NONEXISTENT_KEY", "default_val")
    assert result == "default_val"


def test_get_env_none_default():
    """Test get_env returns None when not set and no default."""
    result = get_env("NONEXISTENT_KEY")
    assert result is None


def test_get_mcp_server_name():
    """Test get_mcp_server_name function."""
    assert get_mcp_server_name() == "scitex-dataset"


def test_get_mcp_instructions():
    """Test MCP instructions contain key information."""
    instructions = get_mcp_instructions()
    assert "OpenNeuro" in instructions
    assert "DANDI" in instructions
    assert "PhysioNet" in instructions
    assert "Zenodo" in instructions
    assert "dataset_openneuro_fetch" in instructions
    assert "dataset_zenodo_fetch" in instructions
    assert "dataset_db_build" in instructions


# EOF
