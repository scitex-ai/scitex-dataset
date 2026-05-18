"""Smoke tests: quickstart example must run to completion (offline-safe)."""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
QUICKSTART = EXAMPLES_DIR / "quickstart.py"


def test_quickstart_example_file_exists_on_disk():
    # Arrange
    path = QUICKSTART
    # Act
    exists = path.exists()
    # Assert
    assert exists, f"missing example: {path}"


def test_quickstart_runs_with_zero_exit_code(tmp_path):
    # Arrange
    cmd = [sys.executable, str(QUICKSTART)]
    # Act
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    # Assert
    assert result.returncode == 0, (
        f"{QUICKSTART.name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
