#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_cli/test__agentic_corebench_prepare.py

"""CLI-level regression for #106: ``ai-for-science corebench prepare``.

The CLI forwarded ``download_full`` to ``corebench.prepare``, whose strict
signature rejects it, so EVERY invocation exited 1 while the Python API
worked. These tests drive the real CLI (a child interpreter running the
console entry point, and click's runner over the real command tree) against
a SYNTHETIC two-capsule raw/ tree in a temp dataset root. No network.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from scitex_dataset._cli import main


def _record(cid: str, question: str, value: float) -> dict:
    return {
        "capsule_id": cid,
        "language": "Python",
        "field": "synthetic",
        "task_prompt": "Run code/main.py.",
        "results": [{question: value}],
    }


def _write_targz(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, body in {"code/main.py": "print('hi')\n", "ReadMe": "x\n"}.items():
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A temp dataset root holding a minimal synthetic corebench raw/ tree."""
    root = tmp_path / "dataset"
    raw = root / "ai-for-science" / "corebench" / "raw"
    (raw / "dataset").mkdir(parents=True)
    (raw / "capsules").mkdir()
    (raw / "dataset" / "core_train.json").write_text(
        json.dumps([_record("capsule-1111111", "Synthetic score?", 0.5123)])
    )
    (raw / "core_test.json").write_text(
        json.dumps([_record("capsule-2222222", "Synthetic mean?", 4.2)])
    )
    for cid in ("capsule-1111111", "capsule-2222222"):
        _write_targz(raw / "capsules" / f"{cid}.tar.gz")
    return root


def _args(root: Path, *extra: str) -> list[str]:
    return [
        "ai-for-science",
        "corebench",
        "prepare",
        "--skip-download",
        "--dataset-root",
        str(root),
        *extra,
    ]


def test_console_entry_point_prepare_exits_zero(dataset_root):
    # Arrange — a child interpreter runs the installed entry point function.
    code = "import sys; from scitex_dataset._cli import main; sys.exit(main())"
    cmd = [sys.executable, "-c", code, *_args(dataset_root)]
    # Act
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # Assert
    assert proc.returncode == 0, proc.stderr


def test_prepare_writes_mapper(dataset_root):
    # Arrange
    runner = CliRunner()
    # Act
    runner.invoke(main, _args(dataset_root))
    # Assert
    mapper = (
        dataset_root / "ai-for-science" / "corebench" / "for_solver" / "index.jsonl"
    )
    assert len(mapper.read_text().splitlines()) == 2


def test_prepare_with_full_flag_exits_zero(dataset_root):
    # Arrange — corebench has no preview split; --full is its default set.
    runner = CliRunner()
    # Act
    result = runner.invoke(main, _args(dataset_root, "--full"))
    # Assert
    assert result.exit_code == 0, result.output


def test_prepare_json_reports_manifest(dataset_root):
    # Arrange
    runner = CliRunner()
    # Act
    result = runner.invoke(main, _args(dataset_root, "--json"))
    # Assert
    assert Path(json.loads(result.output)["manifest"]).is_file()


def test_prepare_does_not_swallow_errors(tmp_path):
    # Arrange — an empty root still fails loud (no raw/ staged).
    runner = CliRunner()
    # Act
    result = runner.invoke(main, _args(tmp_path / "empty"))
    # Assert
    assert result.exit_code == 1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
