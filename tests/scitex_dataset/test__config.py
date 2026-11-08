"""Runtime creation must keep regenerable data outside version control."""

import os
import subprocess
import sys
from pathlib import Path


def test_runtime_creation_ignores_new_cache_in_real_git_repository(tmp_path):
    # Arrange
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".scitex" / "dataset").mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    script = (
        "from scitex_dataset._config import runtime_dir; "
        "(runtime_dir() / 'cache.bin').write_bytes(b'cache')"
    )

    # Act
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".scitex/dataset/runtime/cache.bin"],
        cwd=tmp_path,
        check=False,
    )

    # Assert
    assert ignored.returncode == 0


def test_runtime_creation_preserves_existing_policy_and_cache(tmp_path):
    # Arrange
    runtime = tmp_path / ".scitex" / "dataset" / "runtime"
    runtime.mkdir(parents=True)
    policy = b"# Operator policy\n*\n"
    (runtime / ".gitignore").write_bytes(policy)
    (runtime / "cache.bin").write_bytes(b"existing cache")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    # Act
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from scitex_dataset._config import runtime_dir; runtime_dir()",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    # Assert
    assert (
        (runtime / ".gitignore").read_bytes(),
        (runtime / "cache.bin").read_bytes(),
    ) == (
        policy,
        b"existing cache",
    )
