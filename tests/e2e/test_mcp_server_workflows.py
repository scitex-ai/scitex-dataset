#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/e2e/test_mcp_server_workflows.py

"""End-to-end workflows against real local subsystems (loopback only).

Slow; gated by ``RUN_E2E=1`` and skipped by default. No network: the
MCP tool tree and the skills catalogue are exercised in-process via
the CLI in a child process with an isolated ``SCITEX_DIR``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUN_E2E = os.environ.get("RUN_E2E") == "1"

# Module-level gate: the whole layer is skipped unless RUN_E2E=1.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not RUN_E2E, reason="e2e: set RUN_E2E=1 to run"),
]

CLI = Path(sys.executable).parent / "scitex-dataset"


def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a child process with hermetic env + cwd (see smoke)."""
    env = dict(os.environ)
    env["SCITEX_DIR"] = str(tmp_path / ".scitex")
    env["SCITEX_DATASET_CONFIG"] = " "
    return subprocess.run(
        [str(CLI), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
        check=False,
    )


def test_mcp_list_tools_reports_registered_tool_tree(tmp_path):
    # Arrange
    args = ["mcp", "list-tools"]
    # Act
    proc = _run(args, tmp_path)
    # Assert
    assert proc.returncode == 0 and "Tools:" in proc.stdout, (
        proc.stdout,
        proc.stderr,
    )


def test_skills_list_reports_skill_catalogue(tmp_path):
    # Arrange
    args = ["skills", "list"]
    # Act
    proc = _run(args, tmp_path)
    # Assert
    assert proc.returncode == 0 and "01_installation" in proc.stdout, (
        proc.stdout,
        proc.stderr,
    )
