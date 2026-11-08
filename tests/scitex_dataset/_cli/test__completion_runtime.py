"""Completion installs must keep their runtime files outside version control."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _invocation(tmp_path, command):
    scope = tmp_path / "scitex"
    env = dict(os.environ)
    env["SCITEX_DIR"] = str(scope)
    env["HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    argv = [sys.executable, "-m", "scitex_dataset", *command, "--yes"]
    return argv, env, scope / "dataset" / "runtime" / ".gitignore"


@pytest.mark.parametrize(
    "command", [("completion", "install"), ("install-shell-completion",)]
)
def test_completion_install_creates_runtime_policy(tmp_path, command):
    # Arrange
    argv, env, policy = _invocation(tmp_path, command)
    # Act
    completed = subprocess.run(argv, env=env, cwd=tmp_path, capture_output=True)
    # Assert
    assert (completed.returncode, policy.read_text() if policy.exists() else None) == (
        0,
        "*\n",
    )


@pytest.mark.parametrize(
    "command", [("completion", "install"), ("install-shell-completion",)]
)
def test_completion_repeat_repairs_missing_runtime_policy(tmp_path, command):
    # Arrange
    argv, env, policy = _invocation(tmp_path, command)
    subprocess.run(argv, env=env, cwd=tmp_path, check=True, capture_output=True)
    policy.unlink(missing_ok=True)
    # Act
    completed = subprocess.run(argv, env=env, cwd=tmp_path, capture_output=True)
    # Assert
    assert (completed.returncode, policy.read_text() if policy.exists() else None) == (
        0,
        "*\n",
    )
