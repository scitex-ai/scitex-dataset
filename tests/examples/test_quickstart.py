#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-dataset/tests/examples/test_quickstart.py

"""Smoke test for examples/quickstart.py.

Per scitex-dev audit-project PS303: every example must have a matching
test under tests/examples/. Validates the example parses cleanly. The
full end-to-end execution is covered by tests/scitex_dataset/test_examples.py.
"""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "quickstart.py"


def test_example_file_exists_on_disk():
    # Arrange
    path = EXAMPLE
    # Act
    exists = path.exists()
    # Assert
    assert exists, f"missing example: {path}"


def test_example_compiles_under_py_compile():
    # Arrange
    cmd = [sys.executable, "-m", "py_compile", str(EXAMPLE)]
    # Act
    result = subprocess.run(cmd, check=False)
    # Assert
    assert result.returncode == 0, f"py_compile failed for {EXAMPLE}"
