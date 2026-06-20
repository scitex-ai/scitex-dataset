#!/usr/bin/env python3
"""Tests for ai_for_science.corebench standardize + inventory + download.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The network
download uses the module-level ``corebench._http_download`` helper, which
we replace with a hand-rolled stub via attribute save/restore for the
duration of each test.
"""

import json
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import corebench

# ---------------------------------------------------------------------------
# Network seam — swap corebench._http_download (no unittest.mock)
# ---------------------------------------------------------------------------


@contextmanager
def _swap_http_download(replacement):
    """Replace ``corebench._http_download`` for the duration of the block."""
    saved = corebench._http_download
    corebench._http_download = replacement  # type: ignore[assignment]
    try:
        yield
    finally:
        corebench._http_download = saved  # type: ignore[assignment]


class _HttpRecorder:
    """Records fetch calls and writes deterministic bytes to each dest."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append((url, str(dest)))
        Path(dest).write_bytes(b"capsule-content")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_train_record():
    return {
        "capsule_id": "capsule-1111111",
        "language": "Python",
        "field": "biology",
        "task_prompt": "Read the README.",
        "capsule_title": "A Paper About Stuff",
        "capsule_doi": "10.1234/foo",
        "results": [
            {"What is the AUC?": 0.81},
            {"How many trials?": 12},
            {"What pH?": 7.4},
        ],
    }


def _make_test_record():
    return {
        "capsule_id": "capsule-2222222",
        "language": "R",
        "field": "ecology",
        "task_prompt": "Run the R script.",
        "capsule_title": "Another Paper",
        "capsule_doi": "10.5678/bar",
        "results": [
            {"What is the mean?": 4.2},
            {"What is the variance?": 0.5},
        ],
    }


def _write_capsule_targz(path: Path, cid: str) -> None:
    """Write a minimal real ``.tar.gz`` for capsule ``cid`` at ``path``.

    The per-capsule materializer EXTRACTS each capsule's archive, so the
    test fixtures must be genuine tarballs (not placeholder text).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / f"_scratch_{cid}"
    (scratch / "code").mkdir(parents=True, exist_ok=True)
    (scratch / "code" / "main.py").write_text(f"# {cid}\nprint('hi')\n")
    (scratch / "ReadMe").write_text(f"{cid} readme\n")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(scratch / "code" / "main.py", arcname="code/main.py")
        tf.add(scratch / "ReadMe", arcname="ReadMe")


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Lay out raw/ with oracle JSONs + real capsule tarballs to extract."""
    base = tmp_path / "ai-for-science" / "corebench" / "raw"
    (base / "dataset").mkdir(parents=True)
    (base / "dataset" / "core_train.json").write_text(
        json.dumps([_make_train_record()])
    )
    (base / "core_test.json").write_text(json.dumps([_make_test_record()]))
    (base / "capsules").mkdir()
    _write_capsule_targz(
        base / "capsules" / "capsule-1111111.tar.gz", "capsule-1111111"
    )
    _write_capsule_targz(
        base / "capsules" / "capsule-2222222.tar.gz", "capsule-2222222"
    )
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
# standardize — for_solver tasks (leak-safe uniform schema)
# ---------------------------------------------------------------------------


class TestStandardizeForSolver:
    def test_standardize_writes_index_mapper(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "index.jsonl").is_file()

    def test_index_maps_native_id_to_friendly_id(self, staged_raw_dir):
        # Arrange — capsule-1111111 sorts first, so it gets capsule-001.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        index = _read_jsonl(root / "for_solver" / "index.jsonl")
        row = next(r for r in index if r["native_id"] == "capsule-1111111")
        # Assert
        assert row["friendly_id"] == "capsule-001"

    def test_standardize_writes_per_capsule_task_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "capsule-001" / "task.jsonl").is_file()

    def test_tasks_have_exactly_uniform_keys(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Assert
        assert all(set(t) == {"task_id", "benchmark", "prompt", "data"} for t in tasks)

    def test_tasks_carry_no_answer_value(self, staged_raw_dir):
        # Arrange — the oracle answer 0.81 must not appear in any task row.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        raw_text = (root / "for_solver" / "capsule-001" / "task.jsonl").read_text()
        # Assert
        assert "0.81" not in raw_text

    def test_task_id_uses_difficulty_tier(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        ids = _all_task_ids(root / "for_solver")
        # Assert
        assert "corebench/capsule-1111111__hard__q0" in ids

    def test_task_data_points_at_extracted_input(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
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
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        extracted = root / "for_solver" / "capsule-001" / "input" / "code" / "main.py"
        # Assert
        assert extracted.is_file()

    def test_task_prompt_appends_question_text(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Act
        hard = next(t for t in tasks if t["task_id"].endswith("__hard__q0"))
        # Assert
        assert hard["prompt"].endswith("Question: What is the AUC?")

    def test_standardize_writes_submission_schema_per_capsule(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (
            root / "for_solver" / "capsule-001" / "submission.schema.json"
        ).is_file()

    def test_standardize_writes_capsule_readme(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "capsule-001" / "README.md").is_file()

    def test_only_native_id_materializes_single_capsule(self, staged_raw_dir):
        # Arrange — request only the test-split capsule by native id.
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
            only="capsule-2222222",
        )
        # Assert — capsule-2222222 sorts second → capsule-002 only.
        assert (root / "for_solver" / "capsule-002").is_dir() and not (
            root / "for_solver" / "capsule-001"
        ).exists()

    def test_standardize_does_not_expose_oracle_dataset_dir(self, staged_raw_dir):
        # Arrange — the answer-bearing ``dataset`` dir must never appear.
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert not (root / "for_solver" / "dataset").exists()

    def test_standardize_raises_when_oracle_train_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty-raw"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            corebench.standardize(
                raw_dir=bare,
                for_solver_dir=tmp_path / "fs",
                eval_dir=tmp_path / "ev",
            )


# ---------------------------------------------------------------------------
# standardize — eval answers (operator view)
# ---------------------------------------------------------------------------


class TestStandardizeEval:
    def test_standardize_writes_answers_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "answers.jsonl").is_file()

    def test_answer_task_ids_match_task_ids(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
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

    def test_answers_carry_value_payload(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        # Act
        hard = next(a for a in answers if a["task_id"].endswith("__hard__q0"))
        # Assert
        assert hard["answer"] == {"value": 0.81}

    def test_standardize_writes_evaluate_py(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "evaluate.py").is_file()


class TestEvaluatePyRoundTrip:
    def test_correct_submission_scores_one(self, staged_raw_dir):
        # Arrange — build a submission that matches every oracle value.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        sub = [
            {"task_id": a["task_id"], "answer": a["answer"]["value"]} for a in answers
        ]
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

    def test_wrong_submission_scores_below_one(self, staged_raw_dir):
        # Arrange — every answer deliberately wrong.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        sub = [{"task_id": a["task_id"], "answer": -99999} for a in answers]
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
# build_inventory — writes to for_solver
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def test_build_inventory_writes_inventory_json_in_for_solver(self, staged_raw_dir):
        # Arrange
        for_solver_dir = staged_raw_dir.parent / "for_solver"
        # Act
        corebench.build_inventory(raw_dir=staged_raw_dir, for_solver_dir=for_solver_dir)
        # Assert
        assert (for_solver_dir / "inventory.json").is_file()

    def test_build_inventory_summary_counts_one_train_capsule(self, staged_raw_dir):
        # Arrange
        for_solver_dir = staged_raw_dir.parent / "for_solver"
        # Act
        result = corebench.build_inventory(
            raw_dir=staged_raw_dir, for_solver_dir=for_solver_dir
        )
        # Assert
        assert result["summary"]["n_capsules_train"] == 1


# ---------------------------------------------------------------------------
# download — checksum-verified skip
# ---------------------------------------------------------------------------


class TestDownloadChecksumSkip:
    def test_first_run_fetches_each_capsule(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_fetched"] == 2

    def test_second_run_default_skips_by_existence(self, tmp_path):
        # Arrange — default policy skips present files with NO hashing.
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_have"] == 2

    def test_second_run_skips_verified_when_opt_in(self, tmp_path):
        # Arrange — verify_integrity re-checks sha256 against the ledger.
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], verify_integrity=True
            )
        # Assert
        assert result["n_skipped_verified"] == 2

    def test_second_run_fetches_nothing(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_fetched"] == 0

    def test_tampered_capsule_is_refetched_under_verify(self, tmp_path):
        # Arrange — corrupt a verified file so its sha drifts (only the
        # opt-in integrity pass detects it; default skip would not).
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        (raw_dir / "capsules" / "111.tar.gz").write_bytes(b"TAMPERED")
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], verify_integrity=True
            )
        # Assert
        assert result["n_remismatch"] == 1

    def test_force_refetches_existing(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], force=True
            )
        # Assert
        assert result["n_fetched"] == 2

    def test_download_writes_checksums_ledger(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111"])
        # Assert
        assert (raw_dir / ".checksums.json").is_file()

    def test_download_raises_when_capsule_ids_none_and_oracle_missing(self, tmp_path):
        # Arrange
        bare_raw = tmp_path / "empty-raw"
        bare_raw.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            corebench.download(raw_dir=bare_raw)


# ---------------------------------------------------------------------------
# prepare orchestrator
# ---------------------------------------------------------------------------


def _paths_for(staged_raw_dir):
    from scitex_dataset.ai_for_science import _base

    root = staged_raw_dir.parent
    return _base.BenchmarkPaths(
        benchmark=corebench.BENCHMARK,
        root=root,
        raw_dir=staged_raw_dir,
        for_solver_dir=root / "for_solver",
        eval_dir=root / "eval",
        manifest_dir=root / ".scitex" / "dataset",
    )


class TestPrepare:
    def test_prepare_skip_download_emits_manifest_yaml(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert Path(result["manifest"]).is_file()

    def test_prepare_skip_download_has_standardize_key(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert "standardize" in result

    def test_prepare_with_download_emits_expected_keys(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        rec = _HttpRecorder()
        # Act
        with _swap_http_download(rec):
            result = corebench.prepare(paths=paths, skip_download=False)
        # Assert
        assert {"download", "inventory", "standardize", "manifest"} <= set(result)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
