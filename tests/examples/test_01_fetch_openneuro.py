#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-dataset/tests/examples/test_01_fetch_openneuro.py

"""Smoke test for examples/01_fetch_openneuro.py.

Per scitex-dev audit-project PS303: every example must have a matching
test under tests/examples/. Validates the example parses cleanly without
performing real network fetches.
"""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "01_fetch_openneuro.py"


def test_example_exists():
    assert EXAMPLE.exists(), f"missing example: {EXAMPLE}"


def test_compiles():
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(EXAMPLE)],
        check=True,
    )
