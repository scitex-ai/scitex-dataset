#!/usr/bin/env python3
"""Tests for ai_for_science.biomysterybench standardize + download (no network).

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``.
``snapshot_download`` is imported lazily inside ``download(...)`` from
``huggingface_hub``, so we swap a hand-rolled stub module into
``sys.modules['huggingface_hub']`` for the duration of the test.
"""

import io
import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import biomysterybench

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
            (dst / "snapshot_marker.txt").write_text(repo_id)
        return str(local_dir)


def _hf_stub(recorder):
    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = recorder  # type: ignore[attr-defined]
    return stub


def _build_csv_text(rows: list[dict]) -> str:
    """Render ``rows`` to CSV text without touching the filesystem.

    Keeps the fixture from acquiring an external resource via ``open()``
    (PA-307 STX-TQ005) by writing to an in-memory buffer.
    """
    import csv as _csv

    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _sample_rows():
    return [
        {
            "id": "p1",
            "question": "Which organism is the model?",
            "allowed_domains": "biology",
            "human_solvable": "true",
            "answer_rubric": "Award full marks for Homo sapiens",
        },
        {
            "id": "p2",
            "question": "What gene is upregulated?",
            "allowed_domains": "biology",
            "human_solvable": "true",
            "answer_rubric": "Award full marks for BRCA1",
        },
    ]


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) leaves raw/."""
    base = tmp_path / "ai-for-science" / "biomysterybench" / "raw"
    base.mkdir(parents=True)
    (base / "problems.csv").write_text(_build_csv_text(_sample_rows()))
    return base


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# standardize — for_solver tasks + eval answers (rubric mode)
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_writes_tasks_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "tasks.jsonl").is_file()

    def test_tasks_have_exactly_uniform_keys(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "tasks.jsonl")
        # Assert
        assert all(set(t) == {"task_id", "benchmark", "prompt", "data"} for t in tasks)

    def test_tasks_carry_no_rubric_text(self, staged_raw_dir):
        # Arrange — the rubric "BRCA1" must not leak into for_solver tasks.
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        raw_text = (root / "for_solver" / "tasks.jsonl").read_text()
        # Assert
        assert "BRCA1" not in raw_text

    def test_task_id_prefixes_benchmark(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        ids = {t["task_id"] for t in _read_jsonl(root / "for_solver" / "tasks.jsonl")}
        # Assert
        assert "biomysterybench/p1" in ids

    def test_data_is_null_when_no_environment(self, staged_raw_dir):
        # Arrange — no raw/data dir staged, so each task's data is null.
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "tasks.jsonl")
        # Assert
        assert all(t["data"] is None for t in tasks)

    def test_data_points_at_zip_when_environment_present(self, staged_raw_dir):
        # Arrange — stage raw/data/p1.zip so p1's task data is set.
        (staged_raw_dir / "data").mkdir()
        (staged_raw_dir / "data" / "p1.zip").write_text("zip-bytes")
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "tasks.jsonl")
        p1 = next(t for t in tasks if t["task_id"] == "biomysterybench/p1")
        # Assert
        assert p1["data"] == "./data/p1.zip"

    def test_data_dir_symlinked_into_for_solver(self, staged_raw_dir):
        # Arrange
        (staged_raw_dir / "data").mkdir()
        root = staged_raw_dir.parent
        # Act
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        link = root / "for_solver" / "data"
        assert link.is_symlink() and link.resolve() == (staged_raw_dir / "data")

    def test_oracle_csv_not_symlinked(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert not (root / "for_solver" / "problems.csv").exists()

    def test_answer_ids_match_task_ids(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        biomysterybench.standardize(
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

    def test_answers_carry_rubric(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        # Act
        p1 = next(a for a in answers if a["task_id"] == "biomysterybench/p1")
        # Assert
        assert p1["answer"]["rubric"] == "Award full marks for Homo sapiens"

    def test_evaluate_py_written(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "evaluate.py").is_file()

    def test_raises_when_problems_csv_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            biomysterybench.standardize(
                raw_dir=bare,
                for_solver_dir=tmp_path / "fs",
                eval_dir=tmp_path / "ev",
            )


class TestEvaluatePyRubricMode:
    def test_rubric_mode_marks_tasks_for_grading(self, staged_raw_dir):
        # Arrange — rubric mode is not auto-scorable; each task is flagged.
        root = staged_raw_dir.parent
        biomysterybench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        sub = [{"task_id": "biomysterybench/p1", "answer": "Homo sapiens"}]
        (root / "sub.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(root / "eval" / "evaluate.py"),
                "--submission",
                str(root / "sub.json"),
                "--answers",
                str(root / "eval" / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["n_scored"] == 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestDownload:
    def test_download_preview_returns_preview_snapshot_in_pulled_list(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = biomysterybench.download(raw_dir=raw_dir)
        # Assert
        assert result["snapshots_pulled"] == [biomysterybench.HF_REPO_ID_PREVIEW]

    def test_download_full_pulls_preview_and_full(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = biomysterybench.download(raw_dir=raw_dir, download_full=True)
        # Assert
        assert result["snapshots_pulled"] == [
            biomysterybench.HF_REPO_ID_PREVIEW,
            biomysterybench.HF_REPO_ID_FULL,
        ]


class TestPrepareWithDownload:
    def test_prepare_emits_download_standardize_manifest_keys(self, staged_raw_dir):
        # Arrange — CSV already staged; snapshot_download is a no-op so the
        # staged raw_dir survives the download step.
        from scitex_dataset.ai_for_science import _base

        rec = _SnapshotRecorder(write_dummy=False)
        root = staged_raw_dir.parent
        paths = _base.BenchmarkPaths(
            benchmark=biomysterybench.BENCHMARK,
            root=root,
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
            manifest_dir=root / ".scitex" / "dataset",
        )
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = biomysterybench.prepare(paths=paths, skip_download=False)
        # Assert
        assert {"download", "standardize", "manifest"} <= set(result)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
