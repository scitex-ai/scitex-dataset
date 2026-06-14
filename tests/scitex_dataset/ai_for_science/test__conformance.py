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


# EOF
