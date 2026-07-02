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

    def test_force_reextracts_a_clobbered_capsule(self, materialized):
        # Arrange — clobber an extracted file, then force re-extract.
        _, for_solver, raw, tasks, _ = materialized
        victim = for_solver / "capsule-001" / "input" / "code" / "main.py"
        victim.write_text("TAMPERED\n")
        # Act
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw, force=True
        )
        # Assert
        assert "print('a')" in victim.read_text()


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


# EOF
