#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/smoke/test_cli_happy_path.py

"""Subprocess CLI happy-path smoke tests (fast, <60s, runs on every PR).

Drives the installed ``scitex-dataset`` console script in a child
process with an isolated ``SCITEX_DIR`` — no network, no user state.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

CLI = Path(sys.executable).parent / "scitex-dataset"


def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a child process with hermetic env + cwd.

    Isolation is passed explicitly via ``env=`` / ``cwd=`` (no global
    ``os.environ`` mutation, no ``monkeypatch``): ``SCITEX_DIR`` points
    at the per-test tmp dir, and ``SCITEX_DATASET_CONFIG`` is set BLANK
    (a deleted var gets repopulated from a real ``.env`` via dotenv —
    blank never resolves to a file).
    """
    env = dict(os.environ)
    env["SCITEX_DIR"] = str(tmp_path / ".scitex")
    env["SCITEX_DATASET_CONFIG"] = " "
    return subprocess.run(
        [str(CLI), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env=env,
        check=False,
    )


def test_cli_help_exits_zero(tmp_path):
    # Arrange
    args = ["--help"]
    # Act
    proc = _run(args, tmp_path)
    # Assert
    assert proc.returncode == 0, proc.stderr


def test_cli_help_shows_usage_line(tmp_path):
    # Arrange
    args = ["--help"]
    # Act
    proc = _run(args, tmp_path)
    # Assert
    assert "Usage: scitex-dataset" in proc.stdout, proc.stdout


def test_cli_version_reports_package_version(tmp_path):
    # Arrange
    args = ["--version"]
    # Act
    proc = _run(args, tmp_path)
    # Assert
    assert proc.returncode == 0 and "scitex-dataset" in proc.stdout, (
        proc.stdout,
        proc.stderr,
    )
