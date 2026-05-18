#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/test__branding.py

"""Tests for branding module.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The single
test that needs to set an env var swaps ``os.environ[...]`` directly
via a save/restore context manager.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from scitex_dataset._branding import (
    ENV_PREFIX,
    MCP_SERVER_NAME,
    PACKAGE_NAME,
    get_env,
    get_mcp_instructions,
    get_mcp_server_name,
)


@contextmanager
def _swap_env(key: str, value: str) -> Iterator[None]:
    """Set ``os.environ[key] = value`` for the duration of the block."""
    saved = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


def test_package_name_constant_equals_scitex_dataset():
    # Arrange
    expected = "scitex-dataset"
    # Act
    actual = PACKAGE_NAME
    # Assert
    assert actual == expected


def test_env_prefix_constant_equals_scitex_dataset_uppercase():
    # Arrange
    expected = "SCITEX_DATASET"
    # Act
    actual = ENV_PREFIX
    # Assert
    assert actual == expected


def test_mcp_server_name_constant_equals_scitex_dataset():
    # Arrange
    expected = "scitex-dataset"
    # Act
    actual = MCP_SERVER_NAME
    # Assert
    assert actual == expected


def test_get_env_returns_value_for_prefixed_env_var():
    # Arrange
    key = "SCITEX_DATASET_TEST_KEY"
    # Act
    with _swap_env(key, "test_value"):
        result = get_env("TEST_KEY")
    # Assert
    assert result == "test_value"


def test_get_env_returns_default_when_var_unset():
    # Arrange
    expected = "default_val"
    # Act
    result = get_env("NONEXISTENT_KEY_FOR_BRANDING_TEST", "default_val")
    # Assert
    assert result == expected


def test_get_env_returns_none_when_var_unset_without_default():
    # Arrange
    key = "NONEXISTENT_KEY_FOR_BRANDING_TEST"
    # Act
    result = get_env(key)
    # Assert
    assert result is None


def test_get_mcp_server_name_returns_scitex_dataset():
    # Arrange
    expected = "scitex-dataset"
    # Act
    actual = get_mcp_server_name()
    # Assert
    assert actual == expected


def test_mcp_instructions_mention_openneuro():
    # Arrange
    needle = "OpenNeuro"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_dandi():
    # Arrange
    needle = "DANDI"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_physionet():
    # Arrange
    needle = "PhysioNet"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_zenodo():
    # Arrange
    needle = "Zenodo"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_openneuro_fetch_tool():
    # Arrange
    needle = "openneuro_fetch"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_zenodo_fetch_tool():
    # Arrange
    needle = "zenodo_fetch"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


def test_mcp_instructions_mention_db_build_tool():
    # Arrange
    needle = "db_build"
    # Act
    instructions = get_mcp_instructions()
    # Assert
    assert needle in instructions


# EOF
