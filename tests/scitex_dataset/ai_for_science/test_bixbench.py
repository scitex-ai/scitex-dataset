#!/usr/bin/env python3
"""Tests for ai_for_science.bixbench standardize + download (no network).

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``.
``snapshot_download`` is imported lazily inside ``download(...)`` from
``huggingface_hub``, so we swap a hand-rolled stub module into
``sys.modules['huggingface_hub']`` for the duration of the test.

``standardize`` writes the PER-CAPSULE ``for_solver/`` layout: a root
``index.jsonl`` mapper plus one self-contained ``capsule-NNN/`` dir per
native capsule (holding the EXTRACTED archive under ``input/``). The
fixtures therefore stage a REAL ``.zip`` archive at the path each record's
``data`` points at, so the materializer can extract it.
"""

import json
import subprocess
import sys
import types
import zipfile
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
        # question_id is UNIQUE per question (``<short_id>-qN``) and is the
        # real task key; short_id alone is capsule-scoped.
        "question_id": "sh1-q1",
        "data_folder": "CapsuleFolder-abc.zip",
        "canary": "CANARY-SENTINEL",
        # Oracle fields that must stay out of the for_solver view:
        "answer": "ASXL1",
        "ideal": "gene X",
    }


def _write_capsule_zip(path: Path) -> None:
    """Write a REAL ``.zip`` capsule archive at ``path``.

    The per-capsule materializer EXTRACTS each record's ``data`` archive,
    so the fixture must be a genuine zip (not a bare directory). A
    ``results/`` entry is included to exercise the answer-leak strip — it
    must NOT survive into the extracted ``input/``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("notebook/analysis.py", "# scaffold\nprint('hi')\n")
        zf.writestr("data/matrix.csv", "a,b\n1,2\n")
        # Authors' original output — an answer leak that must be stripped.
        zf.writestr("results/summary.txt", "the answer is here\n")


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) would leave raw/."""
    base = tmp_path / "ai-for-science" / "bixbench" / "raw"
    base.mkdir(parents=True)
    (base / "BixBench.jsonl").write_text(json.dumps(_sample_record()) + "\n")
    _write_capsule_zip(base / "CapsuleFolder-abc.zip")
    return base


def _multi_question_records():
    """Two questions of ONE real capsule: SAME short_id + data_folder, but
    DISTINCT question_id (``bix-1-q1``, ``bix-1-q2``).

    This is the real-schema shape that regressed: BixBench.jsonl has 205
    question rows across only 54 short_ids, so keying task_id on short_id
    collapsed sibling questions. Keying on question_id keeps them distinct
    while ``data_folder`` still groups both into ONE capsule dir.
    """
    return [
        {
            "id": "a32935af",
            "question": "What is the adjusted p-value?",
            "short_id": "bix-1",
            "question_id": "bix-1-q1",
            "data_folder": "CapsuleFolder-shared.zip",
            "answer": "0.0002",
            "ideal": "0.0002",
        },
        {
            "id": "e40e8b38",
            "question": "What is the fold change?",
            "short_id": "bix-1",
            "question_id": "bix-1-q2",
            "data_folder": "CapsuleFolder-shared.zip",
            "answer": "1.9E-05",
            "ideal": "1.9E-05",
        },
    ]


@pytest.fixture
def multi_question_raw_dir(tmp_path):
    """Stage a single capsule that carries TWO distinct questions."""
    base = tmp_path / "ai-for-science" / "bixbench" / "raw"
    base.mkdir(parents=True)
    lines = "\n".join(json.dumps(r) for r in _multi_question_records()) + "\n"
    (base / "BixBench.jsonl").write_text(lines)
    _write_capsule_zip(base / "CapsuleFolder-shared.zip")
    return base


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _all_task_ids(for_solver_dir):
    """Collect task_ids across every per-capsule ``task.jsonl``."""
    ids = set()
    for task_file in for_solver_dir.glob("capsule-*/task.jsonl"):
        for row in _read_jsonl(task_file):
            ids.add(row["task_id"])
    return ids


# ---------------------------------------------------------------------------
# standardize — per-capsule for_solver view + eval answers
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_writes_index_mapper(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "index.jsonl").is_file()

    def test_writes_per_capsule_task_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "capsule-001" / "task.jsonl").is_file()

    def test_index_maps_native_id_to_friendly_id(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        index = _read_jsonl(root / "for_solver" / "index.jsonl")
        row = next(r for r in index if r["native_id"] == "CapsuleFolder-abc")
        # Assert
        assert row["friendly_id"] == "capsule-001"

    def test_tasks_have_exactly_uniform_keys(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
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
        raw_text = (root / "for_solver" / "capsule-001" / "task.jsonl").read_text()
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
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Assert — keyed on the UNIQUE question_id, not the capsule short_id.
        assert tasks[0]["task_id"] == "bixbench/sh1-q1"

    def test_task_id_keyed_on_question_id_not_short_id(self, staged_raw_dir):
        # Arrange — short_id "sh1" is capsule-scoped; question_id "sh1-q1" is
        # the unique per-question key. The task_id must use question_id so
        # sibling questions never collapse onto the same id.
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        task_ids = {
            t["task_id"]
            for t in _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        }
        # Assert — keyed on question_id ("sh1-q1"), never the bare short_id.
        assert task_ids == {"bixbench/sh1-q1"}

    def test_task_data_points_at_extracted_input(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Act
        data_values = {t["data"] for t in tasks}
        # Assert
        assert data_values == {"./input"}

    def test_capsule_archive_extracted_into_input(self, staged_raw_dir):
        # Arrange — the archive is extracted into capsule-001/input/, not a
        # symlink to raw/ (the old flat layout). Scaffold files survive.
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        extracted = (
            root / "for_solver" / "capsule-001" / "input" / "notebook" / "analysis.py"
        )
        # Assert
        assert extracted.is_file()

    def test_leak_dir_stripped_from_input(self, staged_raw_dir):
        # Arrange — the authors' results/ output must not ship to the agent.
        root = staged_raw_dir.parent
        # Act
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert not (
            root / "for_solver" / "capsule-001" / "input" / "results"
        ).exists()

    def test_oracle_manifest_not_exposed(self, staged_raw_dir):
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
        task_ids = _all_task_ids(root / "for_solver")
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


class TestMultiQuestionCapsule:
    """One capsule, two questions (shared short_id + data_folder, distinct
    question_id) must stay two joined tasks — the regression this fix closes."""

    def test_two_questions_yield_two_distinct_task_ids(self, multi_question_raw_dir):
        # Arrange
        root = multi_question_raw_dir.parent
        bixbench.standardize(
            raw_dir=multi_question_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        task_ids = _all_task_ids(root / "for_solver")
        # Assert — NOT collapsed onto a single short_id-keyed task.
        assert task_ids == {"bixbench/bix-1-q1", "bixbench/bix-1-q2"}

    def test_both_questions_land_in_one_capsule_dir(self, multi_question_raw_dir):
        # Arrange — shared data_folder ⇒ one native capsule ⇒ one capsule dir.
        root = multi_question_raw_dir.parent
        bixbench.standardize(
            raw_dir=multi_question_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        capsule_dirs = sorted((root / "for_solver").glob("capsule-*"))
        # Assert
        assert [p.name for p in capsule_dirs] == ["capsule-001"]

    def test_capsule_task_jsonl_carries_both_questions(self, multi_question_raw_dir):
        # Arrange
        root = multi_question_raw_dir.parent
        bixbench.standardize(
            raw_dir=multi_question_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        rows = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Assert — both questions in the SAME capsule's task.jsonl.
        assert {r["task_id"] for r in rows} == {
            "bixbench/bix-1-q1",
            "bixbench/bix-1-q2",
        }

    def test_index_lists_both_task_ids_for_capsule(self, multi_question_raw_dir):
        # Arrange
        root = multi_question_raw_dir.parent
        bixbench.standardize(
            raw_dir=multi_question_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        index = _read_jsonl(root / "for_solver" / "index.jsonl")
        row = next(r for r in index if r["friendly_id"] == "capsule-001")
        # Assert
        assert set(row["task_ids"]) == {"bixbench/bix-1-q1", "bixbench/bix-1-q2"}

    def test_eval_answers_carry_both_task_ids(self, multi_question_raw_dir):
        # Arrange — for_solver + eval stay joined on the question_id key.
        root = multi_question_raw_dir.parent
        bixbench.standardize(
            raw_dir=multi_question_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        answer_ids = {
            a["task_id"] for a in _read_jsonl(root / "eval" / "answers.jsonl")
        }
        # Assert
        assert answer_ids == {"bixbench/bix-1-q1", "bixbench/bix-1-q2"}


class TestEvaluatePyRoundTrip:
    def test_correct_string_submission_scores_one(self, staged_raw_dir):
        # Arrange — string mode: submit the exact oracle answer.
        root = staged_raw_dir.parent
        bixbench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        sub = [{"task_id": "bixbench/sh1-q1", "answer": "ASXL1"}]
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
        sub = [{"task_id": "bixbench/sh1-q1", "answer": "totally wrong"}]
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
        # Arrange — JSONL + capsule zip already staged; snapshot_download is
        # a no-op so the staged raw_dir survives the download step.
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
