"""Regression test: importing tests/conftest.py must not touch site-packages."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"


def test_conftest_leaves_site_packages_untouched():
    # Arrange
    child_code = textwrap.dedent(
        """\
        import os
        import runpy
        import site

        os.environ["COVERAGE_PROCESS_START"] = "SENTINEL_PROCESS_START"
        os.environ["COVERAGE_FILE"] = "SENTINEL_COVERAGE_FILE"

        dirs = list(site.getsitepackages())
        try:
            usersite = site.getusersitepackages()
        except Exception:
            usersite = None
        if usersite and usersite not in dirs:
            dirs.append(usersite)
        before = {}
        for d in dirs:
            try:
                before[d] = set(os.listdir(d))
            except OSError:
                before[d] = None

        runpy.run_path(%r)

        for d, names in before.items():
            try:
                after = set(os.listdir(d))
            except OSError:
                after = None
            assert after == names, f"site-packages dir modified: {d}"
        assert os.environ.get("COVERAGE_PROCESS_START") == "SENTINEL_PROCESS_START"
        assert os.environ.get("COVERAGE_FILE") == "SENTINEL_COVERAGE_FILE"
        """
        % str(CONFTEST_PATH)
    )
    # Act
    proc = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True)
    # Assert
    assert proc.returncode == 0, proc.stderr
