#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-dataset/tests/examples/test_03_cli_usage.py

"""Smoke test for examples/03_cli_usage.sh.

Per scitex-dev audit-project PS303: every example must have a matching
test under tests/examples/. For shell examples, a syntax check (`bash -n`)
proves the script parses cleanly without executing commands.
"""

import os
import subprocess
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "03_cli_usage.sh"


def test_example_shell_script_exists_on_disk():
    # Arrange
    path = EXAMPLE
    # Act
    exists = path.exists()
    # Assert
    assert exists, f"missing example: {path}"


def test_example_shell_script_has_executable_bit():
    # Arrange
    path = EXAMPLE
    # Act
    is_executable = os.access(path, os.X_OK)
    # Assert
    assert is_executable, f"not executable: {path}"


def test_example_shell_script_parses_under_bash_n():
    # Arrange
    cmd = ["bash", "-n", str(EXAMPLE)]
    # Act
    result = subprocess.run(cmd, check=False)
    # Assert
    assert result.returncode == 0, f"bash -n failed for {EXAMPLE}"
