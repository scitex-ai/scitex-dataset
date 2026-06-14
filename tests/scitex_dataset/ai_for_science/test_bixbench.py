#!/usr/bin/env python3
"""Tests for ai_for_science.bixbench standardize + download (no network).

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``.
``snapshot_download`` is imported lazily inside ``download(...)`` from
``huggingface_hub``, so we swap a hand-rolled stub module into
``sys.modules['huggingface_hub']`` for the duration of the test.
"""

import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import bixbench

# ---------------------------------------------------------------------------
# Network seam — hand-rolled huggingface_hub stub (no unittest.mock)
# ---------------------------------------------------------------------------


@contextmanager
def _swap_module(name: str, replacement):
    """Swap ``sys.modules[name]`` with ``replacement`` for the block."""
    saved = sys.modules.get(name)
    sys.modules[name] = replacement
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


class _SnapshotRecorder:
    """Records snapshot_download calls and writes a dummy file per pull."""

    def __init__(self, *, write_dummy=True):
        self.calls = []
        self._write_dummy = write_dummy

    def __call__(self, *, repo_id, repo_type, local_dir, max_workers, token):
        self.calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "local_dir": local_dir,
                "max_workers": max_workers,
                "token": token,
            }
        )
        if self._write_dummy:
            dst = Path(local_dir)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "BixBench.jsonl").write_text("{}\n")
        return str(local_dir)


def _hf_stub(recorder):
    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = recorder  # type: ignore[attr-defined]
    return stub


def _sample_record():
    return {
        "id": "rec-1",
        "question": "Which gene rises?",
        "hypothesis": "Truncating ASXL1 alters expression.",
        "short_id": "sh1",
        "data_folder": "CapsuleFolder-abc",
        "canary": "CANARY-SENTINEL",
        # Oracle fields that must stay out of the for_solver view:
        "answer": "ASXL1",
        "ideal": "gene X",
    }


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) would leave raw/."""
    base = tmp_path / "ai-for-science" / "bixbench" / "raw"
    base.mkdir(parents=True)
    (base / "BixBench.jsonl").write_text(json.dumps(_sample_record()) + "\n")
    (base / "CapsuleFolder-abc").mkdir()
    return base


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# standardize — for_solver tasks + eval answers
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_writes_tasks_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "tasks.jsonl").is_file()

    def test_tasks_have_exactly_uniform_keys(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "tasks.jsonl")
        # Assert
        assert all(set(t) == {"task_id", "benchmark", "prompt", "data"} for t in tasks)

    def test_tasks_carry_no_answer_text(self, staged_raw_dir):
        # Arrange — the oracle answer "ASXL1" must not leak into tasks.
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        raw_text = (root / "for_solver" / "tasks.jsonl").read_text()
        # Assert
        assert "ASXL1" not in raw_text

    def test_task_id_prefixes_benchmark(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "tasks.jsonl")
        # Assert
        assert tasks[0]["task_id"] == "bixbench/sh1"

    def test_data_folder_symlinked_into_for_solver(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        link = root / "for_solver" / "CapsuleFolder-abc"
        assert link.is_symlink() and link.resolve() == (
            staged_raw_dir / "CapsuleFolder-abc"
        )

    def test_oracle_manifest_not_symlinked(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert not (root / "for_solver" / "BixBench.jsonl").exists()

    def test_answer_ids_match_task_ids(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        task_ids = {
            t["task_id"] for t in _read_jsonl(root / "for_solver" / "tasks.jsonl")
        }
        # Act
        answer_ids = {
            a["task_id"] for a in _read_jsonl(root / "eval" / "answers.jsonl")
        }
        # Assert
        assert answer_ids == task_ids

    def test_evaluate_py_written(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "evaluate.py").is_file()

    def test_raises_when_raw_jsonl_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            bixbench.standardize(
                raw_dir=bare,
                for_solver_dir=tmp_path / "fs",
                eval_dir=tmp_path / "ev",
            )


class TestEvaluatePyRoundTrip:
    def test_correct_string_submission_scores_one(self, staged_raw_dir):
        # Arrange — string mode: submit the exact oracle answer.
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        sub = [{"task_id": "bixbench/sh1", "answer": "ASXL1"}]
        (root / "good.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(root / "eval" / "evaluate.py"),
                "--submission",
                str(root / "good.json"),
                "--answers",
                str(root / "eval" / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["score"] == 1.0

    def test_wrong_string_submission_scores_below_one(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        sub = [{"task_id": "bixbench/sh1", "answer": "totally wrong"}]
        (root / "bad.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(root / "eval" / "evaluate.py"),
                "--submission",
                str(root / "bad.json"),
                "--answers",
                str(root / "eval" / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["score"] < 1.0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestDownload:
    def test_download_returns_repo_id_in_pulled_list(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.download(raw_dir=raw_dir)
        # Assert
        assert result["snapshots_pulled"] == [bixbench.HF_REPO_ID]

    def test_download_records_resolved_set(self, tmp_path):
        # Arrange — snapshot_download natively skips by etag/sha; we just
        # record the resolved snapshot path it returns.
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.download(raw_dir=raw_dir)
        # Assert
        assert result["resolved"] == str(raw_dir)


class TestPrepareWithDownload:
    def test_prepare_emits_download_standardize_manifest_keys(self, staged_raw_dir):
        # Arrange — JSONL already staged; snapshot_download is a no-op so
        # the staged raw_dir survives the download step.
        from scitex_dataset.ai_for_science import _base

        rec = _SnapshotRecorder(write_dummy=False)
        root = staged_raw_dir.parent
        paths = _base.BenchmarkPaths(
            benchmark=bixbench.BENCHMARK,
            root=root,
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
            manifest_dir=root / ".scitex" / "dataset",
        )
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.prepare(paths=paths, skip_download=False)
        # Assert
        assert {"download", "standardize", "manifest"} <= set(result)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
