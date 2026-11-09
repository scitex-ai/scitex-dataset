"""Original release guards must reject bad inputs before store access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / ".github/ci/run_release_tests.py"
WORKFLOW = ROOT / ".github/workflows/pypi-publish-and-github-release-on-tag.yml"


def _invoke(directory, version):
    environment = {"PATH": os.defpath, "HOME": str(directory), "LANG": "C.UTF-8"}
    return subprocess.run(
        [sys.executable, str(DRIVER), "--python-version", version],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_wrong_interpreter_is_refused_before_store_access(tmp_path):
    # Arrange
    (tmp_path / "tests").mkdir()
    # Act
    result = _invoke(tmp_path, "0.0")
    # Assert
    assert result.returncode != 0 and "interpreter does not match" in result.stderr


def test_missing_suite_is_refused_before_store_access(tmp_path):
    # Arrange
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Act
    result = _invoke(tmp_path, version)
    # Assert
    assert result.returncode != 0 and "tests directory is missing" in result.stderr


def test_missing_store_cannot_use_a_fleet_default(tmp_path):
    # Arrange
    (tmp_path / "tests").mkdir()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Act
    result = _invoke(tmp_path, version)
    # Assert
    assert result.returncode != 0 and "explicit job-owned store" in result.stderr


def test_main_release_deadline_is_preserved():
    # Arrange
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # Act
    deadline = workflow["jobs"]["test"]["timeout-minutes"]
    # Assert
    assert deadline == 30


@pytest.mark.parametrize("job", ["test", "build"])
def test_setup_python_output_is_bound_to_the_interpreter_guard(job):
    # Arrange
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]
    setup = next(step for step in steps if step.get("id") == "setup-python")
    guard = next(
        step
        for step in steps
        if step.get("name") == "Qualify the requested interpreter"
    )
    # Act
    binding = (setup["uses"], guard["env"]["RELEASE_SETUP_PYTHON"])
    # Assert
    assert binding == (
        "actions/setup-python@v5",
        "${{ steps.setup-python.outputs.python-path }}",
    )
