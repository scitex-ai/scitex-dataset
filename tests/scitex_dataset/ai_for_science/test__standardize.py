#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/ai_for_science/test__conformance.py

"""Conformance test — a NEW agentic benchmark drops into the contract.

Generalization guarantee: adding a benchmark (e.g. AstaBench) means
writing one small adapter that yields a uniform task list + answer list
+ the names of its answer-free problem data. Feed those to the shared
``_standardize`` writers and the result MUST be: a leak-clean
``for_solver/`` (uniform schema, no oracle reachable) and an ``eval/``
whose ``evaluate.py`` scores a correct submission 1.0.

This exercises the shared machinery via a synthetic 4th benchmark
("astabench") so the promise "a new one just works" is checked in CI —
no real download, no benchmark-specific code under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _standardize

BENCH = "astabench"


def _build_synthetic_benchmark(tmp_path: Path):
    """Stand up a standardized 'astabench' from the shared helpers alone."""
    root = tmp_path / BENCH
    raw = root / "raw"
    raw.mkdir(parents=True)
    # Agent-visible problem data + an answer-bearing oracle that must
    # never reach for_solver/ (it is NOT named in data_links).
    (raw / "problem_q1.txt").write_text("analyze this dataset\n")
    (raw / "ORACLE_answers.csv").write_text("q1,42\n")

    tasks = [
        {
            "task_id": f"{BENCH}/q1",
            "benchmark": BENCH,
            "prompt": "What is the answer?",
            "data": "./problem_q1.txt",
        }
    ]
    answers = [{"task_id": f"{BENCH}/q1", "answer": {"value": 42.0}, "meta": {}}]

    for_solver = root / "for_solver"
    eval_dir = root / "eval"
    _standardize.write_for_solver(
        for_solver_dir=for_solver,
        tasks=tasks,
        raw_dir=raw,
        data_links=["problem_q1.txt"],  # allow-list: only the problem data
    )
    _standardize.write_eval(
        eval_dir=eval_dir,
        answers=answers,
        evaluate_py_source=_standardize.render_evaluate_py("numeric"),
    )
    return root, for_solver, eval_dir


@pytest.fixture
def built(tmp_path):
    return _build_synthetic_benchmark(tmp_path)


class TestNewBenchmarkConformance:
    def test_tasks_have_uniform_schema(self, built):
        # Arrange
        _, for_solver, _ = built
        # Act
        rec = json.loads((for_solver / "tasks.jsonl").read_text().splitlines()[0])
        # Assert
        assert set(rec) == set(_standardize.TASK_KEYS)

    def test_tasks_carry_no_answer_field(self, built):
        # Arrange
        _, for_solver, _ = built
        # Act
        rec = json.loads((for_solver / "tasks.jsonl").read_text().splitlines()[0])
        # Assert
        assert "answer" not in rec

    def test_problem_data_is_symlinked(self, built):
        # Arrange
        _, for_solver, _ = built
        # Act
        link = for_solver / "problem_q1.txt"
        # Assert
        assert link.is_symlink()

    def test_oracle_not_reachable_from_for_solver(self, built):
        # Arrange
        _, for_solver, _ = built
        # Act
        leaked = (for_solver / "ORACLE_answers.csv").exists()
        # Assert
        assert not leaked

    def test_submission_schema_emitted(self, built):
        # Arrange
        _, for_solver, _ = built
        # Act
        schema = for_solver / "submission.schema.json"
        # Assert
        assert schema.is_file()

    def test_eval_answers_keyed_by_same_task_id(self, built):
        # Arrange
        _, for_solver, eval_dir = built
        # Act
        tasks = {
            json.loads(line)["task_id"]
            for line in (for_solver / "tasks.jsonl").read_text().splitlines()
        }
        answers = {
            json.loads(line)["task_id"]
            for line in (eval_dir / "answers.jsonl").read_text().splitlines()
        }
        # Assert
        assert tasks == answers

    def test_evaluate_py_is_emitted(self, built):
        # Arrange
        _, _, eval_dir = built
        # Act
        evaluate = eval_dir / "evaluate.py"
        # Assert
        assert evaluate.is_file()

    def test_correct_submission_scores_one(self, built):
        # Arrange
        root, _, eval_dir = built
        sub = [{"task_id": f"{BENCH}/q1", "answer": 42.0}]
        (root / "good.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(eval_dir / "evaluate.py"),
                "--submission",
                str(root / "good.json"),
                "--answers",
                str(eval_dir / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["score"] == 1.0


# ---------------------------------------------------------------------------
# Per-capsule materializer — friendly ids + mapper + extracted input/
# ---------------------------------------------------------------------------


def _make_archive(path: Path, members: dict[str, str]) -> None:
    """Write a real archive at ``path`` (suffix-dispatched) with members.

    ``members`` maps an in-archive relative path to its text content. The
    suffix selects the writer: ``.tar.gz`` via :mod:`tarfile`, ``.zip``
    via :mod:`zipfile`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith((".tar.gz", ".tgz", ".tar")):
        scratch = path.parent / f"_scratch_{path.name}"
        scratch.mkdir(parents=True, exist_ok=True)
        for rel, text in members.items():
            f = scratch / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        with tarfile.open(path, "w:gz") as tf:
            for rel in members:
                tf.add(scratch / rel, arcname=rel)
    elif path.name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as zf:
            for rel, text in members.items():
                zf.writestr(rel, text)
    else:  # pragma: no cover — test helper guard
        raise ValueError(f"unsupported archive suffix: {path}")


def _build_two_capsule_benchmark(tmp_path: Path):
    """Stand up a 2-capsule benchmark with real archives under raw/capsules."""
    root = tmp_path / "twocap"
    raw = root / "raw"
    caps = raw / "capsules"
    # Native ids deliberately out of insertion order so the friendly-id
    # SORT (not insertion order) is what gets tested.
    _make_archive(
        caps / "capsule-bbb222.tar.gz",
        {"code/main.py": "print('b')\n", "ReadMe": "capsule b\n"},
    )
    _make_archive(
        caps / "capsule-aaa111.tar.gz",
        {"code/main.py": "print('a')\n", "ReadMe": "capsule a\n"},
    )

    # Two tasks for capsule-bbb222 (multi-task), one for capsule-aaa111.
    tasks = [
        {
            "task_id": "twocap/capsule-bbb222__hard__q0",
            "benchmark": "twocap",
            "prompt": "Q B0",
            "data": "./capsules/capsule-bbb222.tar.gz",
        },
        {
            "task_id": "twocap/capsule-bbb222__hard__q1",
            "benchmark": "twocap",
            "prompt": "Q B1",
            "data": "./capsules/capsule-bbb222.tar.gz",
        },
        {
            "task_id": "twocap/capsule-aaa111__hard__q0",
            "benchmark": "twocap",
            "prompt": "Q A0",
            "data": "./capsules/capsule-aaa111.tar.gz",
        },
    ]
    for_solver = root / "for_solver"
    return root, for_solver, raw, tasks


@pytest.fixture
def two_capsule(tmp_path):
    return _build_two_capsule_benchmark(tmp_path)


@pytest.fixture
def materialized(two_capsule):
    """Run the default (all-capsule) materializer once; share its output."""
    root, for_solver, raw, tasks = two_capsule
    result = _standardize.write_for_solver_per_capsule(
        for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
    )
    return root, for_solver, raw, tasks, result


def _read_index(for_solver: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (for_solver / "index.jsonl").read_text().splitlines()
        if line
    ]


def _read_task_jsonl(for_solver: Path, friendly: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (for_solver / friendly / "task.jsonl").read_text().splitlines()
        if line
    ]


class TestFriendlyId:
    def test_friendly_id_first_position_is_capsule_001(self):
        # Arrange
        position = 0
        # Act
        friendly = _standardize.friendly_capsule_id(position)
        # Assert
        assert friendly == "capsule-001"

    def test_friendly_id_zero_pads_to_three_digits(self):
        # Arrange
        position = 11
        # Act
        friendly = _standardize.friendly_capsule_id(position)
        # Assert
        assert friendly == "capsule-012"


class TestBuildCapsuleIndex:
    def test_index_sorts_native_ids_ascending_into_friendly_ids(self, two_capsule):
        # Arrange
        _, _, _, tasks = two_capsule
        # Act
        index = _standardize.build_capsule_index(tasks)
        # Assert — aaa111 sorts before bbb222, so it gets capsule-001.
        assert [(r["friendly_id"], r["native_id"]) for r in index] == [
            ("capsule-001", "capsule-aaa111"),
            ("capsule-002", "capsule-bbb222"),
        ]

    def test_index_groups_all_task_ids_for_one_capsule(self, two_capsule):
        # Arrange
        _, _, _, tasks = two_capsule
        # Act
        index = _standardize.build_capsule_index(tasks)
        bbb = next(r for r in index if r["native_id"] == "capsule-bbb222")
        # Assert
        assert bbb["task_ids"] == [
            "twocap/capsule-bbb222__hard__q0",
            "twocap/capsule-bbb222__hard__q1",
        ]

    def test_index_skips_tasks_with_no_archive_data(self):
        # Arrange — a task whose data is None has nothing to materialize.
        tasks = [{"task_id": "x/t0", "benchmark": "x", "prompt": "p", "data": None}]
        # Act
        index = _standardize.build_capsule_index(tasks)
        # Assert
        assert index == []


class TestPerCapsuleLayout:
    def test_mapper_written_at_for_solver_root(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        present = (for_solver / "index.jsonl").is_file()
        # Assert
        assert present

    def test_capsule_dir_uses_friendly_name(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        present = (for_solver / "capsule-001").is_dir()
        # Assert — aaa111 → capsule-001.
        assert present

    def test_archive_is_extracted_into_input(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        extracted = for_solver / "capsule-001" / "input" / "code" / "main.py"
        # Act
        body = extracted.read_text() if extracted.is_file() else ""
        # Assert
        assert "print('a')" in body

    def test_input_is_a_real_dir_not_a_symlink(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        input_dir = for_solver / "capsule-001" / "input"
        # Act
        is_real = input_dir.is_dir() and not input_dir.is_symlink()
        # Assert
        assert is_real

    def test_input_is_not_left_as_an_archive(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        leftover = (for_solver / "capsule-001" / "capsule-aaa111.tar.gz").exists()
        # Assert
        assert not leftover

    def test_task_jsonl_holds_only_its_own_rows(self, materialized):
        # Arrange — capsule-002 is bbb222 (2 tasks); must not see aaa111.
        _, for_solver, _, _, _ = materialized
        # Act
        ids = {r["task_id"] for r in _read_task_jsonl(for_solver, "capsule-002")}
        # Assert
        assert ids == {
            "twocap/capsule-bbb222__hard__q0",
            "twocap/capsule-bbb222__hard__q1",
        }

    def test_task_data_rewritten_to_input(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        row = _read_task_jsonl(for_solver, "capsule-001")[0]
        # Assert
        assert row["data"] == "./input"

    def test_task_id_stays_canonical_inside_capsule(self, materialized):
        # Arrange — directory uses friendly id; task_id stays native.
        _, for_solver, _, _, _ = materialized
        # Act
        row = _read_task_jsonl(for_solver, "capsule-001")[0]
        # Assert
        assert row["task_id"] == "twocap/capsule-aaa111__hard__q0"

    def test_capsule_keeps_uniform_task_keys(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        row = _read_task_jsonl(for_solver, "capsule-001")[0]
        # Assert
        assert set(row) == set(_standardize.TASK_KEYS)

    def test_schema_copied_into_first_capsule(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        present = (for_solver / "capsule-001" / "submission.schema.json").is_file()
        # Assert
        assert present

    def test_schema_copied_into_second_capsule(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        present = (for_solver / "capsule-002" / "submission.schema.json").is_file()
        # Assert
        assert present

    def test_example_prefilled_with_real_task_ids(self, materialized):
        # Arrange — bbb222 has two tasks; example lists both real ids.
        _, for_solver, _, _, _ = materialized
        example = json.loads(
            (for_solver / "capsule-002" / "submission.example.json").read_text()
        )
        # Act
        ids = [e["task_id"] for e in example]
        # Assert
        assert ids == [
            "twocap/capsule-bbb222__hard__q0",
            "twocap/capsule-bbb222__hard__q1",
        ]

    def test_example_answers_are_placeholders(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        example = json.loads(
            (for_solver / "capsule-002" / "submission.example.json").read_text()
        )
        # Act
        all_placeholder = all(e["answer"] == "<your answer here>" for e in example)
        # Assert
        assert all_placeholder

    def test_readme_names_the_friendly_id(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        readme = (for_solver / "capsule-001" / "README.md").read_text()
        # Assert
        assert "capsule-001" in readme

    def test_readme_lists_the_real_task_id(self, materialized):
        # Arrange
        _, for_solver, _, _, _ = materialized
        # Act
        readme = (for_solver / "capsule-001" / "README.md").read_text()
        # Assert
        assert "twocap/capsule-aaa111__hard__q0" in readme

    def test_no_sibling_capsule_referenced_inside_dir(self, materialized):
        # Arrange — nothing inside capsule-001 may mention the other native id.
        _, for_solver, _, _, _ = materialized
        cap = for_solver / "capsule-001"
        # Act
        blob = "".join(
            p.read_text(errors="ignore") for p in cap.rglob("*") if p.is_file()
        )
        # Assert
        assert "capsule-bbb222" not in blob


class TestOnlyFilter:
    def test_only_friendly_id_materializes_a_single_capsule(self, two_capsule):
        # Arrange
        root, for_solver, raw, tasks = two_capsule
        # Act
        result = _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, only="capsule-002"
        )
        # Assert
        assert result["n_materialized"] == 1

    def test_only_friendly_id_leaves_other_capsule_absent(self, two_capsule):
        # Arrange
        root, for_solver, raw, tasks = two_capsule
        # Act
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, only="capsule-002"
        )
        # Assert
        assert not (for_solver / "capsule-001").exists()

    def test_only_native_id_resolves_via_mapper_to_friendly_dir(self, two_capsule):
        # Arrange — pass the NATIVE id; it must resolve to capsule-001.
        root, for_solver, raw, tasks = two_capsule
        # Act
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, only="capsule-aaa111"
        )
        # Assert
        assert (for_solver / "capsule-001" / "input").is_dir()

    def test_only_still_writes_the_full_mapper(self, two_capsule):
        # Arrange — even with --only, index.jsonl lists every capsule.
        root, for_solver, raw, tasks = two_capsule
        # Act
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, only="capsule-002"
        )
        # Assert
        assert len(_read_index(for_solver)) == 2

    def test_only_unknown_selector_raises_key_error(self, two_capsule):
        # Arrange
        root, for_solver, raw, tasks = two_capsule
        # Act
        # Assert
        with pytest.raises(KeyError):
            _standardize.write_for_solver_per_capsule(
                for_solver_dir=for_solver,
                tasks=tasks,
                raw_dir=raw,
                only="capsule-nope",
            )


class TestIdempotencyAndForce:
    def test_second_run_skips_already_extracted_capsules(self, materialized):
        # Arrange — first run done by the fixture.
        _, for_solver, raw, tasks, _ = materialized
        # Act
        result = _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
        )
        # Assert
        assert result["n_skipped"] == 2

    def test_second_run_materializes_nothing(self, materialized):
        # Arrange
        _, for_solver, raw, tasks, _ = materialized
        # Act
        result = _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
        )
        # Assert
        assert result["n_materialized"] == 0

    @pytest.fixture
    def _case_force_preserves_and_refuses_a_clobbered_capsule(self, materialized):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, for_solver, raw, tasks, _ = materialized
        victim = for_solver / 'capsule-001' / 'input' / 'code' / 'main.py'
        victim.write_text('TAMPERED\n')

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, force=True)
                except _standardize.StaleMaterializationError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, force=True)
            return (victim,)
        return run_case

    def test_force_preserves_and_refuses_a_clobbered_capsule_raises_stale_error(self, _case_force_preserves_and_refuses_a_clobbered_capsule):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_force_preserves_and_refuses_a_clobbered_capsule
        # Act
        ctx = pytest.raises(_standardize.StaleMaterializationError)
        # Assert
        with ctx:
            run_case()

    def test_force_preserves_and_refuses_a_clobbered_capsule_tampered_file_preserved(self, _case_force_preserves_and_refuses_a_clobbered_capsule):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_force_preserves_and_refuses_a_clobbered_capsule
        # Act
        victim, = run_case(capture_error=True)
        # Assert
        assert victim.read_text() == 'TAMPERED\n'


class TestArchiveExtraction:
    def test_missing_archive_for_selected_capsule_raises(self, two_capsule):
        # Arrange — delete the archive the selected capsule needs.
        root, for_solver, raw, tasks = two_capsule
        (raw / "capsules" / "capsule-aaa111.tar.gz").unlink()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            _standardize.write_for_solver_per_capsule(
                for_solver_dir=for_solver,
                tasks=tasks,
                raw_dir=raw,
                only="capsule-001",
            )

    def test_zip_archive_is_extracted_into_input(self, tmp_path):
        # Arrange — a capsule whose archive is a .zip.
        raw = tmp_path / "z" / "raw"
        _make_archive(raw / "capsules" / "capsule-z1.zip", {"hello.txt": "hi\n"})
        tasks = [
            {
                "task_id": "z/capsule-z1__hard__q0",
                "benchmark": "z",
                "prompt": "Q",
                "data": "./capsules/capsule-z1.zip",
            }
        ]
        for_solver = tmp_path / "z" / "for_solver"
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
        )
        extracted = for_solver / "capsule-001" / "input" / "hello.txt"
        # Act
        body = extracted.read_text() if extracted.is_file() else ""
        # Assert
        assert body == "hi\n"


@pytest.fixture
def stripped_notebook_cell(tmp_path):
    """Materialize a capsule whose archive ships an EXECUTED notebook
    (code-cell outputs carry the answer), then return the first cell of the
    agent-visible notebook so each test asserts one facet of the strip."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "outputs": [{"output_type": "stream", "text": "AUC = 0.94\n"}],
                "source": ["print('AUC = 0.94')"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    raw = tmp_path / "nb" / "raw"
    _make_archive(
        raw / "capsules" / "capsule-nb1.zip",
        {"analysis_executed.ipynb": json.dumps(nb)},
    )
    tasks = [
        {
            "task_id": "nb/capsule-nb1__hard__q0",
            "benchmark": "nb",
            "prompt": "Q",
            "data": "./capsules/capsule-nb1.zip",
        }
    ]
    for_solver = tmp_path / "nb" / "for_solver"
    _standardize.write_for_solver_per_capsule(
        for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
    )
    out = json.loads(
        (for_solver / "capsule-001" / "input" / "analysis_executed.ipynb").read_text()
    )
    return out["cells"][0]


class TestNotebookOutputStrip:
    def test_executed_notebook_outputs_emptied(self, stripped_notebook_cell):
        # Arrange
        cell = stripped_notebook_cell
        # Act
        outputs = cell["outputs"]
        # Assert
        assert outputs == []

    def test_executed_notebook_execution_count_reset(self, stripped_notebook_cell):
        # Arrange
        cell = stripped_notebook_cell
        # Act
        execution_count = cell["execution_count"]
        # Assert
        assert execution_count is None

    def test_notebook_code_kept_as_scaffold(self, stripped_notebook_cell):
        # Arrange
        cell = stripped_notebook_cell
        # Act
        source = "".join(cell["source"])
        # Assert
        assert "print('AUC = 0.94')" in source


# ---------------------------------------------------------------------------
# Running-log / output leak strip + value-verify guard (any depth, any cohort)
# ---------------------------------------------------------------------------


def _build_leaky_capsule(tmp_path: Path, *, value_in_code: bool = False):
    """A capsule whose archive ships author run-logs/outputs at several
    depths (nested ``code/dump`` + ``code/log``, top-level ``results/``, and a
    loose ``data/*_log.csv``) plus decoys that must be KEPT (``catalog.csv``,
    ``input.csv``, ``log_utils.py``). The task's answer is ``0.931818``."""
    raw = tmp_path / "leak" / "raw"
    members = {
        "code/main.py": "x = 1\nprint('run')\n",
        "code/log_utils.py": "def log():\n    return 1\n",
        "code/dump/log_evaluate.txt": "eval loss = 1.469021\n",
        "code/log/train_log.txt": "acc = 0.931818\n",
        "results/output.txt": "final = 0.7569591\n",
        "data/feature_selection_log.csv": "score,0.5551234\n",
        "data/catalog.csv": "id,name\n1,foo\n",
        "data/input.csv": "a,b\n1,2\n",
    }
    if value_in_code:
        members["code/main.py"] = "ANSWER = 0.931818\nprint('run')\n"
    _make_archive(raw / "capsules" / "capsule-leak1.tar.gz", members)
    tasks = [
        {
            "task_id": "leak/capsule-leak1__hard__q0",
            "benchmark": "leak",
            "prompt": "Q",
            "data": "./capsules/capsule-leak1.tar.gz",
        }
    ]
    answer_values = {"leak/capsule-leak1__hard__q0": 0.931818}
    for_solver = tmp_path / "leak" / "for_solver"
    return raw, for_solver, tasks, answer_values


@pytest.fixture
def leak_input(tmp_path):
    raw, for_solver, tasks, answer_values = _build_leaky_capsule(tmp_path)
    _standardize.write_for_solver_per_capsule(
        for_solver_dir=for_solver,
        tasks=tasks,
        raw_dir=raw,
        answer_values=answer_values,
    )
    return for_solver / "capsule-001" / "input"


class TestRunningLogLeakStrip:
    def test_nested_dump_dir_removed(self, leak_input):
        # Arrange
        target = leak_input / "code" / "dump"
        # Act
        present = target.exists()
        # Assert
        assert not present

    def test_nested_log_dir_removed(self, leak_input):
        # Arrange
        target = leak_input / "code" / "log"
        # Act
        present = target.exists()
        # Assert
        assert not present

    def test_top_level_results_dir_removed(self, leak_input):
        # Arrange
        target = leak_input / "results"
        # Act
        present = target.exists()
        # Assert
        assert not present

    def test_loose_log_file_in_data_removed(self, leak_input):
        # Arrange
        target = leak_input / "data" / "feature_selection_log.csv"
        # Act
        present = target.exists()
        # Assert
        assert not present

    def test_catalog_csv_is_kept(self, leak_input):
        # Arrange
        target = leak_input / "data" / "catalog.csv"
        # Act
        present = target.is_file()
        # Assert
        assert present

    def test_plain_input_csv_is_kept(self, leak_input):
        # Arrange
        target = leak_input / "data" / "input.csv"
        # Act
        present = target.is_file()
        # Assert
        assert present

    def test_log_named_code_file_is_kept(self, leak_input):
        # Arrange
        target = leak_input / "code" / "log_utils.py"
        # Act
        present = target.is_file()
        # Assert
        assert present

    def test_code_main_is_kept(self, leak_input):
        # Arrange
        target = leak_input / "code" / "main.py"
        # Act
        present = target.is_file()
        # Assert
        assert present

    def test_no_answer_value_survives_anywhere(self, leak_input):
        # Arrange
        files = [p for p in leak_input.rglob("*") if p.is_file()]
        # Act
        blob = "".join(p.read_text(errors="ignore") for p in files)
        # Assert
        assert "0.931818" not in blob


class TestValueLeakGuard:
    def test_guard_masks_value_in_kept_file(self, tmp_path):
        # Arrange
        raw, for_solver, tasks, answer_values = _build_leaky_capsule(
            tmp_path, value_in_code=True
        )
        # Act
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver,
            tasks=tasks,
            raw_dir=raw,
            answer_values=answer_values,
        )
        main = (for_solver / "capsule-001" / "input" / "code" / "main.py").read_text()
        # Assert — value redacted in place; file kept (not raised, not removed).
        assert "0.931818" not in main and _standardize._VALUE_REDACTION in main

    def test_assert_backstop_raises_on_residual_value(self, tmp_path):
        # Arrange — a high-precision value left behind in a text file.
        d = tmp_path / "resid"
        d.mkdir()
        (d / "leftover.txt").write_text("final score = 0.931818\n")
        # Act
        # Assert
        with pytest.raises(_standardize.AnswerLeakError):
            _standardize._assert_no_value_leak(d, [0.931818], capsule="c")

    def test_guard_ignores_low_information_integer_answer(self, tmp_path):
        # Arrange — two top-level dirs so input/ is not de-nested; 1000 is the
        # answer, hard-coded in a KEPT code file.
        raw = tmp_path / "intc" / "raw"
        _make_archive(
            raw / "capsules" / "capsule-int1.tar.gz",
            {"code/main.py": "BATCH = 1000\nprint('x')\n", "data/x.csv": "a\n1\n"},
        )
        tasks = [
            {
                "task_id": "intc/capsule-int1__hard__q0",
                "benchmark": "intc",
                "prompt": "Q",
                "data": "./capsules/capsule-int1.tar.gz",
            }
        ]
        for_solver = tmp_path / "intc" / "for_solver"
        # Act — 1000 is too generic to be a leak signal, so this must not raise.
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver,
            tasks=tasks,
            raw_dir=raw,
            answer_values={"intc/capsule-int1__hard__q0": 1000},
        )
        # Assert
        assert (for_solver / "capsule-001" / "input" / "code" / "main.py").is_file()


def _generated_result(tmp_path, answers, submission, *, mode="numeric"):
    script = tmp_path / "evaluate.py"
    script.write_text(_standardize.render_evaluate_py(mode))
    oracle = tmp_path / "answers.jsonl"
    oracle.write_text("".join(json.dumps(row) + "\n" for row in answers))
    submitted = tmp_path / "submission.json"
    submitted.write_text(json.dumps(submission))
    process = subprocess.run([sys.executable, str(script), "--submission",
                              str(submitted), "--answers", str(oracle)],
                             capture_output=True, text=True, timeout=3)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


class TestGeneratedReferenceContract:
    @pytest.fixture
    def _case_repeated_references_have_one_explicit_ungradeable_question(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        rows = [{'task_id': 'synthetic/q1', 'answer': {'value': value}} for value in (3.1, 3.2, 3.3)]
        rows.append({'task_id': 'synthetic/q2', 'answer': {'value': 5}})
        reference_records = rows
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 3.3}, {'task_id': 'synthetic/q2', 'answer': 5}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    def test_repeated_references_have_one_explicit_ungradeable_question_unique_question_count(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == 2

    def test_repeated_references_have_one_explicit_ungradeable_question_gradeable_correct_count(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert result['n_scored'] == result['n_correct'] == 1

    def test_repeated_references_have_one_explicit_ungradeable_question_ungradeable_count(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert result['n_ungradeable'] == 1

    def test_repeated_references_have_one_explicit_ungradeable_question_unsupported_policy_status(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['status'] == 'needs_reference_policy'

    def test_repeated_references_have_one_explicit_ungradeable_question_all_reference_count(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['n_references'] == 3

    def test_repeated_references_have_one_explicit_ungradeable_question_correctness_absent(self, _case_repeated_references_have_one_explicit_ungradeable_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_repeated_references_have_one_explicit_ungradeable_question
        # Act
        result, = run_case()
        # Assert
        assert 'correct' not in result['per_task'][0]

    @pytest.fixture
    def _case_invalid_reference_is_not_invalid_solver_answer(self, tmp_path, value):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': value}}, {'task_id': 'synthetic/q2', 'answer': {'value': 5}}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 5}, {'task_id': 'synthetic/q2', 'answer': 5}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    @pytest.mark.parametrize('value', [None, True, 'nan', 'inf', 10 ** 400])
    def test_invalid_reference_is_not_invalid_solver_answer_reference_error_status(self, _case_invalid_reference_is_not_invalid_solver_answer, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_reference_is_not_invalid_solver_answer
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['status'] == 'invalid_reference'

    @pytest.mark.parametrize('value', [None, True, 'nan', 'inf', 10 ** 400])
    def test_invalid_reference_is_not_invalid_solver_answer_assigned_and_gradeable_counts(self, _case_invalid_reference_is_not_invalid_solver_answer, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_reference_is_not_invalid_solver_answer
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == 2 and result['n_scored'] == 1

    @pytest.mark.parametrize('value', [None, True, 'nan', 'inf', 10 ** 400])
    def test_invalid_reference_is_not_invalid_solver_answer_valid_peer_correct(self, _case_invalid_reference_is_not_invalid_solver_answer, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_reference_is_not_invalid_solver_answer
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][1]['correct'] is True

    @pytest.fixture
    def _case_invalid_numeric_submission_remains_malformed(self, tmp_path, value):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': 5}}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': value}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    @pytest.mark.parametrize('value', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_remains_malformed_malformed_status(self, _case_invalid_numeric_submission_remains_malformed, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_remains_malformed
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['status'] == 'malformed'

    @pytest.mark.parametrize('value', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_remains_malformed_graded_failure(self, _case_invalid_numeric_submission_remains_malformed, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_remains_malformed
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['correct'] is False

    @pytest.mark.parametrize('value', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_remains_malformed_assigned_and_scored_counts(self, _case_invalid_numeric_submission_remains_malformed, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_remains_malformed
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == 1 and result['n_scored'] == 1

    @pytest.mark.parametrize('value', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_remains_malformed_failed_score(self, _case_invalid_numeric_submission_remains_malformed, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_remains_malformed
        # Act
        result, = run_case()
        # Assert
        assert result['n_correct'] == 0 and result['score'] == 0

    @pytest.fixture
    def _case_missing_numeric_submission_remains_denominator_failure(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': 5}}, {'task_id': 'synthetic/q2', 'answer': {'value': 5}}]
        submission_records = [{'task_id': 'synthetic/q2', 'answer': 5}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    def test_missing_numeric_submission_remains_denominator_failure_missing_status(self, _case_missing_numeric_submission_remains_denominator_failure):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_numeric_submission_remains_denominator_failure
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['status'] == 'no_submission'

    def test_missing_numeric_submission_remains_denominator_failure_graded_failure(self, _case_missing_numeric_submission_remains_denominator_failure):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_numeric_submission_remains_denominator_failure
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['correct'] is False

    def test_missing_numeric_submission_remains_denominator_failure_assigned_scored_count(self, _case_missing_numeric_submission_remains_denominator_failure):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_numeric_submission_remains_denominator_failure
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_scored'] == 2

    def test_missing_numeric_submission_remains_denominator_failure_score_and_gradeability(self, _case_missing_numeric_submission_remains_denominator_failure):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_numeric_submission_remains_denominator_failure
        # Act
        result, = run_case()
        # Assert
        assert result['score'] == 0.5 and result['n_ungradeable'] == 0

    @pytest.fixture
    def _case_ungradeable_assignment_has_no_measured_score(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': 5}}, {'task_id': 'synthetic/q1', 'answer': {'value': 6}}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 5}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    def test_ungradeable_assignment_has_no_measured_score_assigned_ungradeable_count(self, _case_ungradeable_assignment_has_no_measured_score):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_ungradeable_assignment_has_no_measured_score
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_ungradeable'] == 1

    def test_ungradeable_assignment_has_no_measured_score_score_absent(self, _case_ungradeable_assignment_has_no_measured_score):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_ungradeable_assignment_has_no_measured_score
        # Act
        result, = run_case()
        # Assert
        assert result['n_scored'] == 0 and result['score'] is None

    @pytest.fixture
    def _case_single_reference_keeps_scalar_tolerance(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': 1000}}, {'task_id': 'synthetic/q2', 'answer': {'value': 0}}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 1000.5}, {'task_id': 'synthetic/q2', 'answer': 0.0005}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records)
            return (result,)
        return run_case

    def test_single_reference_keeps_scalar_tolerance(self, _case_single_reference_keeps_scalar_tolerance):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_single_reference_keeps_scalar_tolerance
        # Act
        result, = run_case()
        # Assert
        assert result['n_scored'] == result['n_correct'] == 2

    @pytest.fixture
    def _case_duplicate_submission_refuses_instead_of_last_wins(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        script = tmp_path / 'evaluate.py'
        script.write_text(_standardize.render_evaluate_py('numeric'))
        oracle = tmp_path / 'answers.jsonl'
        oracle.write_text(json.dumps({'task_id': 'synthetic/q1', 'answer': {'value': 5}}))
        submitted = tmp_path / 'submission.json'
        submitted.write_text(json.dumps([{'task_id': 'synthetic/q1', 'answer': 0}, {'task_id': 'synthetic/q1', 'answer': 5}]))

        def run_case(*, capture_error=False):
            result = subprocess.run([sys.executable, str(script), '--submission', str(submitted), '--answers', str(oracle)], capture_output=True, text=True, timeout=3)
            return (result,)
        return run_case

    def test_duplicate_submission_refuses_instead_of_last_wins_nonzero_exit(self, _case_duplicate_submission_refuses_instead_of_last_wins):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_duplicate_submission_refuses_instead_of_last_wins
        # Act
        result, = run_case()
        # Assert
        assert result.returncode != 0

    def test_duplicate_submission_refuses_instead_of_last_wins_duplicate_diagnostic(self, _case_duplicate_submission_refuses_instead_of_last_wins):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_duplicate_submission_refuses_instead_of_last_wins
        # Act
        result, = run_case()
        # Assert
        assert 'duplicate submission task_id' in result.stderr

    def test_duplicate_submission_refuses_instead_of_last_wins_stdout_empty(self, _case_duplicate_submission_refuses_instead_of_last_wins):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_duplicate_submission_refuses_instead_of_last_wins
        # Act
        result, = run_case()
        # Assert
        assert not result.stdout


class TestGeneratedStringSubmissionContract:
    @pytest.fixture
    def _case_absent_null_remains_graded_failure(self, tmp_path, submission, status):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': 'none'}}]
        submission_records = submission

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            row = result['per_task'][0]
            return (result, row, status)
        return run_case

    @pytest.mark.parametrize('submission,status', [([], 'no_submission'), ([{'task_id': 'synthetic/q1', 'answer': None}], 'malformed')])
    def test_absent_null_remains_graded_failure_failure_status(self, _case_absent_null_remains_graded_failure, submission, status):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_absent_null_remains_graded_failure
        # Act
        result, row, status = run_case()
        # Assert
        assert row.get('status') == status

    @pytest.mark.parametrize('submission,status', [([], 'no_submission'), ([{'task_id': 'synthetic/q1', 'answer': None}], 'malformed')])
    def test_absent_null_remains_graded_failure_graded_failure(self, _case_absent_null_remains_graded_failure, submission, status):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_absent_null_remains_graded_failure
        # Act
        result, row, status = run_case()
        # Assert
        assert row['correct'] is False

    @pytest.mark.parametrize('submission,status', [([], 'no_submission'), ([{'task_id': 'synthetic/q1', 'answer': None}], 'malformed')])
    def test_absent_null_remains_graded_failure_assigned_scored_count(self, _case_absent_null_remains_graded_failure, submission, status):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_absent_null_remains_graded_failure
        # Act
        result, row, status = run_case()
        # Assert
        assert result['n'] == result['n_scored'] == 1

    @pytest.mark.parametrize('submission,status', [([], 'no_submission'), ([{'task_id': 'synthetic/q1', 'answer': None}], 'malformed')])
    def test_absent_null_remains_graded_failure_correct_ungradeable_count(self, _case_absent_null_remains_graded_failure, submission, status):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_absent_null_remains_graded_failure
        # Act
        result, row, status = run_case()
        # Assert
        assert result['n_correct'] == result['n_ungradeable'] == 0

    @pytest.mark.parametrize('submission,status', [([], 'no_submission'), ([{'task_id': 'synthetic/q1', 'answer': None}], 'malformed')])
    def test_absent_null_remains_graded_failure_failed_score(self, _case_absent_null_remains_graded_failure, submission, status):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_absent_null_remains_graded_failure
        # Act
        result, row, status = run_case()
        # Assert
        assert result['score'] == 0.0

    @pytest.fixture
    def _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy(self, tmp_path, expected, submitted):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': {'value': expected}}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': submitted}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            return (result,)
        return run_case

    @pytest.mark.parametrize('expected,submitted', [('none', 'none'), ('Mixed CASE Answer', '\n mixed  case\tanswer \n'), ('42', '42'), ('true', True), ('42', 42), ('1.5', 1.5), ("['x']", ['x']), ("{'x': 'y'}", {'x': 'y'})])
    def test_nonnull_answer_keeps_original_coercion_case_whitespace_policy_correct_result(self, _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy, expected, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['correct'] is True

    @pytest.mark.parametrize('expected,submitted', [('none', 'none'), ('Mixed CASE Answer', '\n mixed  case\tanswer \n'), ('42', '42'), ('true', True), ('42', 42), ('1.5', 1.5), ("['x']", ['x']), ("{'x': 'y'}", {'x': 'y'})])
    def test_nonnull_answer_keeps_original_coercion_case_whitespace_policy_assigned_scored_correct_count(self, _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy, expected, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_scored'] == result['n_correct'] == 1

    @pytest.mark.parametrize('expected,submitted', [('none', 'none'), ('Mixed CASE Answer', '\n mixed  case\tanswer \n'), ('42', '42'), ('true', True), ('42', 42), ('1.5', 1.5), ("['x']", ['x']), ("{'x': 'y'}", {'x': 'y'})])
    def test_nonnull_answer_keeps_original_coercion_case_whitespace_policy_full_score(self, _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy, expected, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_answer_keeps_original_coercion_case_whitespace_policy
        # Act
        result, = run_case()
        # Assert
        assert result['score'] == 1.0


class TestGeneratedStringReferenceContract:
    @pytest.fixture
    def _case_missing_reference_answer_field_refuses_without_payload(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        script = tmp_path / 'evaluate.py'
        script.write_text(_standardize.render_evaluate_py('string'))
        oracle = tmp_path / 'answers.jsonl'
        sentinel = 'synthetic-private-reference-do-not-print'
        oracle.write_text(json.dumps({'task_id': 'synthetic/q1', 'meta': sentinel}))
        submitted = tmp_path / 'submission.json'
        submitted.write_text(json.dumps([{'task_id': 'synthetic/q1', 'answer': 'none'}]))

        def run_case(*, capture_error=False):
            result = subprocess.run([sys.executable, str(script), '--submission', str(submitted), '--answers', str(oracle)], capture_output=True, text=True, timeout=3)
            return (result, sentinel)
        return run_case

    def test_missing_reference_answer_field_refuses_without_payload_nonzero_exit(self, _case_missing_reference_answer_field_refuses_without_payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_reference_answer_field_refuses_without_payload
        # Act
        result, sentinel = run_case()
        # Assert
        assert result.returncode != 0

    def test_missing_reference_answer_field_refuses_without_payload_missing_field_diagnostic(self, _case_missing_reference_answer_field_refuses_without_payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_reference_answer_field_refuses_without_payload
        # Act
        result, sentinel = run_case()
        # Assert
        assert 'reference record missing answer field' in result.stderr

    def test_missing_reference_answer_field_refuses_without_payload_private_payload_absent(self, _case_missing_reference_answer_field_refuses_without_payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_reference_answer_field_refuses_without_payload
        # Act
        result, sentinel = run_case()
        # Assert
        assert sentinel not in result.stderr

    def test_missing_reference_answer_field_refuses_without_payload_stdout_empty(self, _case_missing_reference_answer_field_refuses_without_payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_reference_answer_field_refuses_without_payload
        # Act
        result, sentinel = run_case()
        # Assert
        assert not result.stdout

    @pytest.fixture
    def _case_present_null_string_reference_is_ungradeable(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': None}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 'none'}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            row = result['per_task'][0]
            return (result, row)
        return run_case

    def test_present_null_string_reference_is_ungradeable_reference_error_status(self, _case_present_null_string_reference_is_ungradeable):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_string_reference_is_ungradeable
        # Act
        result, row = run_case()
        # Assert
        assert row.get('status') == 'invalid_reference'

    def test_present_null_string_reference_is_ungradeable_correctness_absent(self, _case_present_null_string_reference_is_ungradeable):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_string_reference_is_ungradeable
        # Act
        result, row = run_case()
        # Assert
        assert 'correct' not in row

    def test_present_null_string_reference_is_ungradeable_assigned_ungradeable_count(self, _case_present_null_string_reference_is_ungradeable):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_string_reference_is_ungradeable
        # Act
        result, row = run_case()
        # Assert
        assert result['n'] == result['n_ungradeable'] == 1

    def test_present_null_string_reference_is_ungradeable_score_absent(self, _case_present_null_string_reference_is_ungradeable):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_string_reference_is_ungradeable
        # Act
        result, row = run_case()
        # Assert
        assert result['n_scored'] == 0 and result['score'] is None

    @pytest.fixture
    def _case_actual_literal_none_reference_and_answer_remain_correct(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': 'none'}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': 'none'}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            return (result,)
        return run_case

    def test_actual_literal_none_reference_and_answer_remain_correct_correct_result(self, _case_actual_literal_none_reference_and_answer_remain_correct):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_actual_literal_none_reference_and_answer_remain_correct
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['correct'] is True

    def test_actual_literal_none_reference_and_answer_remain_correct_assigned_scored_correct_count(self, _case_actual_literal_none_reference_and_answer_remain_correct):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_actual_literal_none_reference_and_answer_remain_correct
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_scored'] == result['n_correct'] == 1

    def test_actual_literal_none_reference_and_answer_remain_correct_full_score(self, _case_actual_literal_none_reference_and_answer_remain_correct):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_actual_literal_none_reference_and_answer_remain_correct
        # Act
        result, = run_case()
        # Assert
        assert result['score'] == 1.0


class TestGeneratedNullReferenceWrapperContract:
    @pytest.fixture
    def _case_all_null_recognized_wrapper_is_ungradeable(self, tmp_path, payload):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': payload}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': str(payload)}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            return (result,)
        return run_case

    @pytest.mark.parametrize('payload', [{'value': None}, {'answer': None, 'ideal': None}])
    def test_all_null_recognized_wrapper_is_ungradeable_reference_error_status(self, _case_all_null_recognized_wrapper_is_ungradeable, payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_all_null_recognized_wrapper_is_ungradeable
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0].get('status') == 'invalid_reference'

    @pytest.mark.parametrize('payload', [{'value': None}, {'answer': None, 'ideal': None}])
    def test_all_null_recognized_wrapper_is_ungradeable_correctness_absent(self, _case_all_null_recognized_wrapper_is_ungradeable, payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_all_null_recognized_wrapper_is_ungradeable
        # Act
        result, = run_case()
        # Assert
        assert 'correct' not in result['per_task'][0]

    @pytest.mark.parametrize('payload', [{'value': None}, {'answer': None, 'ideal': None}])
    def test_all_null_recognized_wrapper_is_ungradeable_assigned_ungradeable_count(self, _case_all_null_recognized_wrapper_is_ungradeable, payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_all_null_recognized_wrapper_is_ungradeable
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_ungradeable'] == 1

    @pytest.mark.parametrize('payload', [{'value': None}, {'answer': None, 'ideal': None}])
    def test_all_null_recognized_wrapper_is_ungradeable_score_absent(self, _case_all_null_recognized_wrapper_is_ungradeable, payload):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_all_null_recognized_wrapper_is_ungradeable
        # Act
        result, = run_case()
        # Assert
        assert result['n_scored'] == 0 and result['score'] is None

    @pytest.fixture
    def _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved(self, tmp_path, payload, submitted):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        reference_records = [{'task_id': 'synthetic/q1', 'answer': payload}]
        submission_records = [{'task_id': 'synthetic/q1', 'answer': submitted}]

        def run_case(*, capture_error=False):
            result = _generated_result(tmp_path, reference_records, submission_records, mode='string')
            return (result,)
        return run_case

    @pytest.mark.parametrize('payload,submitted', [({'answer': None, 'ideal': 'none'}, 'none'), ({'unrecognized': 'benign-value'}, "{'unrecognized': 'benign-value'}")])
    def test_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved_correct_result(self, _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved, payload, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved
        # Act
        result, = run_case()
        # Assert
        assert result['per_task'][0]['correct'] is True

    @pytest.mark.parametrize('payload,submitted', [({'answer': None, 'ideal': 'none'}, 'none'), ({'unrecognized': 'benign-value'}, "{'unrecognized': 'benign-value'}")])
    def test_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved_assigned_scored_correct_count(self, _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved, payload, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved
        # Act
        result, = run_case()
        # Assert
        assert result['n'] == result['n_scored'] == result['n_correct'] == 1

    @pytest.mark.parametrize('payload,submitted', [({'answer': None, 'ideal': 'none'}, 'none'), ({'unrecognized': 'benign-value'}, "{'unrecognized': 'benign-value'}")])
    def test_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved_full_score(self, _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved, payload, submitted):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_nonnull_fallback_and_nonwrapper_dict_policy_are_preserved
        # Act
        result, = run_case()
        # Assert
        assert result['score'] == 1.0


def _materialized_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class TestMaterializationIdentity:
    @pytest.fixture
    def _case_legacy_root_refuses_before_mapper_or_purge(self, two_capsule, force):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks = two_capsule
        fs.mkdir()
        (fs / 'tasks.jsonl').write_text('historical output\n')
        before = _materialized_bytes(fs)

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, force=force)
                except _standardize.StaleMaterializationError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, force=force)
            return (before, fs)
        return run_case

    @pytest.mark.parametrize('force', [False, True])
    def test_legacy_root_refuses_before_mapper_or_purge_raises_stale_error(self, _case_legacy_root_refuses_before_mapper_or_purge, force):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_legacy_root_refuses_before_mapper_or_purge
        # Act
        ctx = pytest.raises(_standardize.StaleMaterializationError)
        # Assert
        with ctx:
            run_case()

    @pytest.mark.parametrize('force', [False, True])
    def test_legacy_root_refuses_before_mapper_or_purge_historical_bytes_preserved(self, _case_legacy_root_refuses_before_mapper_or_purge, force):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_legacy_root_refuses_before_mapper_or_purge
        # Act
        before, fs = run_case(capture_error=True)
        # Assert
        assert _materialized_bytes(fs) == before

    @pytest.fixture
    def _case_stale_cache_is_preserved_even_force(self, materialized, change):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks, _ = materialized
        identity = None
        references = None
        if change == 'tasks':
            tasks = [{**row, 'prompt': 'changed assignment'} for row in tasks]
        elif change == 'archive':
            _make_archive(raw / 'capsules/capsule-aaa111.tar.gz', {'code/main.py': "print('changed')\n"})
        elif change == 'mapper':
            (fs / 'index.jsonl').write_text('historical mapper changed\n')
        elif change == 'ledger':
            (fs / '.materialization-identity.json').write_text('[]')
        elif change == 'source':
            identity = {'revision': 'changed'}
        else:
            references = {tasks[0]['task_id']: [0.111111]}
        before = _materialized_bytes(fs)

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, force=True, source_identity=identity, reference_values=references)
                except _standardize.StaleMaterializationError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, force=True, source_identity=identity, reference_values=references)
            return (before, fs)
        return run_case

    @pytest.mark.parametrize('change', ['tasks', 'archive', 'mapper', 'ledger', 'source', 'references'])
    def test_stale_cache_is_preserved_even_force_raises_stale_error(self, _case_stale_cache_is_preserved_even_force, change):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_stale_cache_is_preserved_even_force
        # Act
        ctx = pytest.raises(_standardize.StaleMaterializationError)
        # Assert
        with ctx:
            run_case()

    @pytest.mark.parametrize('change', ['tasks', 'archive', 'mapper', 'ledger', 'source', 'references'])
    def test_stale_cache_is_preserved_even_force_historical_bytes_preserved(self, _case_stale_cache_is_preserved_even_force, change):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_stale_cache_is_preserved_even_force
        # Act
        before, fs = run_case(capture_error=True)
        # Assert
        assert _materialized_bytes(fs) == before

    @pytest.fixture
    def _case_changed_friendly_mapping_preserves_existing_capsules(self, materialized):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks, _ = materialized
        changed = [row for row in tasks if 'bbb222' in row['task_id']]
        before = _materialized_bytes(fs)

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=changed, raw_dir=raw, force=True)
                except _standardize.StaleMaterializationError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=changed, raw_dir=raw, force=True)
            return (before, fs)
        return run_case

    def test_changed_friendly_mapping_preserves_existing_capsules_raises_stale_error(self, _case_changed_friendly_mapping_preserves_existing_capsules):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_changed_friendly_mapping_preserves_existing_capsules
        # Act
        ctx = pytest.raises(_standardize.StaleMaterializationError)
        # Assert
        with ctx:
            run_case()

    def test_changed_friendly_mapping_preserves_existing_capsules_historical_bytes_preserved(self, _case_changed_friendly_mapping_preserves_existing_capsules):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_changed_friendly_mapping_preserves_existing_capsules
        # Act
        before, fs = run_case(capture_error=True)
        # Assert
        assert _materialized_bytes(fs) == before

    @pytest.fixture
    def _case_only_does_not_require_unselected_archive(self, two_capsule):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks = two_capsule
        (raw / 'capsules/capsule-bbb222.tar.gz').unlink()

        def run_case(*, capture_error=False):
            result = _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, only='capsule-001')
            return (result,)
        return run_case

    def test_only_does_not_require_unselected_archive(self, _case_only_does_not_require_unselected_archive):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_only_does_not_require_unselected_archive
        # Act
        result, = run_case()
        # Assert
        assert result['n_materialized'] == 1

    @pytest.fixture
    def _case_private_identity_remains_outside_solver_capsules(self, two_capsule):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks = two_capsule
        source = {'revision': 'private-source-identity'}

        def run_case(*, capture_error=False):
            result = _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, source_identity=source)
            ledger_path = Path(result['identity_ledger'])
            return (fs, ledger_path, source)
        return run_case

    def test_private_identity_remains_outside_solver_capsules_ledger_private_mode(self, _case_private_identity_remains_outside_solver_capsules):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_private_identity_remains_outside_solver_capsules
        # Act
        fs, ledger_path, source = run_case()
        # Assert
        assert ledger_path.stat().st_mode & 511 == 384

    def test_private_identity_remains_outside_solver_capsules_ledger_source_identity(self, _case_private_identity_remains_outside_solver_capsules):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_private_identity_remains_outside_solver_capsules
        # Act
        fs, ledger_path, source = run_case()
        # Assert
        assert json.loads(ledger_path.read_text())['source_identity'] == source

    def test_private_identity_remains_outside_solver_capsules_solver_identity_absent(self, _case_private_identity_remains_outside_solver_capsules):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_private_identity_remains_outside_solver_capsules
        # Act
        fs, ledger_path, source = run_case()
        # Assert
        for cap in (fs / 'capsule-001', fs / 'capsule-002'):
            assert b'private-source-identity' not in b''.join(_materialized_bytes(cap).values())

    @pytest.fixture
    def _case_operator_inventory_survives_qualified_reuse(self, materialized):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks, _ = materialized
        inventory = fs / 'inventory.json'
        inventory.write_text('{"operator_metadata": true}\n')

        def run_case(*, capture_error=False):
            result = _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw)
            return (inventory, result)
        return run_case

    def test_operator_inventory_survives_qualified_reuse_qualified_skip_count(self, _case_operator_inventory_survives_qualified_reuse):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_operator_inventory_survives_qualified_reuse
        # Act
        inventory, result = run_case()
        # Assert
        assert result['n_skipped'] == 2

    def test_operator_inventory_survives_qualified_reuse_inventory_bytes_preserved(self, _case_operator_inventory_survives_qualified_reuse):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_operator_inventory_survives_qualified_reuse
        # Act
        inventory, result = run_case()
        # Assert
        assert inventory.read_text() == '{"operator_metadata": true}\n'

    @pytest.fixture
    def _case_invalid_masking_reference_refuses_before_outputs(self, two_capsule, value):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks = two_capsule
        tid = next((row['task_id'] for row in tasks if 'aaa111' in row['task_id']))

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, reference_values={tid: [value]})
                except _standardize.InvalidReferenceSourceError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, reference_values={tid: [value]})
            return (fs,)
        return run_case

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), 10 ** 400])
    def test_invalid_masking_reference_refuses_before_outputs_raises_reference_error(self, _case_invalid_masking_reference_refuses_before_outputs, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_masking_reference_refuses_before_outputs
        # Act
        ctx = pytest.raises(_standardize.InvalidReferenceSourceError)
        # Assert
        with ctx:
            run_case()

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), 10 ** 400])
    def test_invalid_masking_reference_refuses_before_outputs_outputs_absent(self, _case_invalid_masking_reference_refuses_before_outputs, value):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_masking_reference_refuses_before_outputs
        # Act
        fs, = run_case(capture_error=True)
        # Assert
        assert not fs.exists()


class TestPublicAnswerSchemaAndReferenceSamples:
    @pytest.fixture
    def _case_schema_is_derived_only_from_public_answer_type(self):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tasks = [{'task_id': 'synthetic/q1', 'answer_type': 'number'}, {'task_id': 'synthetic/q2', 'answer_type': 'list'}, {'task_id': 'synthetic/q3'}]

        def run_case(*, capture_error=False):
            schema = _standardize.submission_schema_for_tasks(tasks)
            return (schema,)
        return run_case

    def test_schema_is_derived_only_from_public_answer_type_explicit_type_count(self, _case_schema_is_derived_only_from_public_answer_type):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_schema_is_derived_only_from_public_answer_type
        # Act
        schema, = run_case()
        # Assert
        assert len(schema['items']['allOf']) == 2

    def test_schema_is_derived_only_from_public_answer_type_number_type(self, _case_schema_is_derived_only_from_public_answer_type):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_schema_is_derived_only_from_public_answer_type
        # Act
        schema, = run_case()
        # Assert
        assert schema['items']['allOf'][0]['then']['properties']['answer']['type'] == ['number', 'null']

    def test_schema_is_derived_only_from_public_answer_type_list_type(self, _case_schema_is_derived_only_from_public_answer_type):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_schema_is_derived_only_from_public_answer_type
        # Act
        schema, = run_case()
        # Assert
        assert schema['items']['allOf'][1]['then']['properties']['answer']['type'] == ['array', 'null']

    def test_schema_is_derived_only_from_public_answer_type_undeclared_type_unconstrained(self, _case_schema_is_derived_only_from_public_answer_type):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_schema_is_derived_only_from_public_answer_type
        # Act
        schema, = run_case()
        # Assert
        assert schema['items']['properties']['answer'] == {}

    @pytest.fixture
    def _case_bad_declaration_refuses_before_output(self, two_capsule):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        _, fs, raw, tasks = two_capsule
        tasks[0]['answer_type'] = 'derived-from-oracle'

        def run_case(*, capture_error=False):
            if capture_error:
                try:
                    _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw)
                except ValueError as captured:
                    pass
            else:
                _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw)
            return (fs,)
        return run_case

    def test_bad_declaration_refuses_before_output_raises_declaration_error(self, _case_bad_declaration_refuses_before_output):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_bad_declaration_refuses_before_output
        # Act
        ctx = pytest.raises(ValueError, match='answer_type')
        # Assert
        with ctx:
            run_case()

    def test_bad_declaration_refuses_before_output_outputs_absent(self, _case_bad_declaration_refuses_before_output):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_bad_declaration_refuses_before_output
        # Act
        fs, = run_case(capture_error=True)
        # Assert
        assert not fs.exists()

    @pytest.fixture
    def _case_all_reference_samples_reach_existing_value_guard(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        raw, fs, tasks, _ = _build_leaky_capsule(tmp_path)
        tid = tasks[0]['task_id']

        def run_case(*, capture_error=False):
            _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, reference_values={tid: [0.111111, 0.931818]})
            contents = b''.join(_materialized_bytes(fs / 'capsule-001/input').values())
            return (contents,)
        return run_case

    def test_all_reference_samples_reach_existing_value_guard(self, _case_all_reference_samples_reach_existing_value_guard):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_all_reference_samples_reach_existing_value_guard
        # Act
        contents, = run_case()
        # Assert
        assert b'0.931818' not in contents

    @pytest.fixture
    def _case_list_valued_answer_is_one_reference_sample(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        raw = tmp_path / 'raw'
        fs = tmp_path / 'solver'
        value = ['distinctive-ref-one', 'distinctive-ref-two']
        _make_archive(raw / 'capsule-list.tar.gz', {'code/main.py': str(value)})
        tasks = [{'task_id': 'synthetic/q1', 'benchmark': 'synthetic', 'prompt': 'p', 'data': './capsule-list.tar.gz'}]

        def run_case(*, capture_error=False):
            _standardize.write_for_solver_per_capsule(for_solver_dir=fs, tasks=tasks, raw_dir=raw, reference_values={'synthetic/q1': [value]})
            body = (fs / 'capsule-001/input/main.py').read_text()
            return (body, value)
        return run_case

    def test_list_valued_answer_is_one_reference_sample_reference_absent(self, _case_list_valued_answer_is_one_reference_sample):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_list_valued_answer_is_one_reference_sample
        # Act
        body, value = run_case()
        # Assert
        assert str(value) not in body

    def test_list_valued_answer_is_one_reference_sample_redaction_marker_present(self, _case_list_valued_answer_is_one_reference_sample):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_list_valued_answer_is_one_reference_sample
        # Act
        body, value = run_case()
        # Assert
        assert _standardize._VALUE_REDACTION in body


# EOF
