#!/usr/bin/env python3
"""Reference-field contract between each adapter and the scorers (#115).

The adapter that writes ``eval/answers.jsonl`` declares ``REFERENCE_FIELD``;
both scorers (``score_submission`` and the generated ``evaluate.py``) must
grade against exactly that key. #115: BixBench's upstream ``answer`` column
is a BOOLEAN hypothesis flag and ``ideal`` is the reference, but the scorer
picked ``answer`` first, so every BixBench task scored wrong.

No mocks: synthetic upstream records shaped like each source's published
schema (field names + types), real ``standardize`` runs on tmp dirs, real
scorer calls. All reference values below are invented.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import biomysterybench, bixbench, corebench
from scitex_dataset.ai_for_science._reference import parse_interval, reference_fields
from scitex_dataset.ai_for_science._score import OracleIntegrityError, score_submission

# ---------------------------------------------------------------------------
# Schema-shaped synthetic upstream sources
# ---------------------------------------------------------------------------


def _bixbench_record(question_id: str, ideal, *, answer=True, eval_mode="str_verifier"):
    """One row shaped like ``futurehouse/BixBench`` (``answer`` is a bool)."""
    rec = {
        "id": f"id-{question_id}",
        "tag": "synthetic",
        "version": "1.5",
        "question": f"Synthetic question {question_id}?",
        "distractors": ["d1", "d2", "d3"],
        "canary": "CANARY-SYNTHETIC",
        "capsule_uuid": "00000000-0000-0000-0000-000000000000",
        "short_id": "syn-1",
        "question_id": question_id,
        "hypothesis": "A synthetic hypothesis.",
        "result": "A synthetic result.",
        "answer": answer,
        "categories": "Synthetic",
        "paper": "",
        "data_folder": None,
        "eval_mode": eval_mode,
    }
    if ideal is not _ABSENT:
        rec["ideal"] = ideal
    return rec


_ABSENT = object()


def _stage_bixbench(tmp_path: Path, records: list[dict]) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / bixbench.ORACLE_MANIFEST_NAME).write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return raw


def _standardize_bixbench(tmp_path: Path, records: list[dict]) -> Path:
    raw = _stage_bixbench(tmp_path, records)
    bixbench.standardize(
        raw_dir=raw, for_solver_dir=tmp_path / "for_solver", eval_dir=tmp_path / "eval"
    )
    return tmp_path / "eval"


def _standardize_biomysterybench(tmp_path: Path, rubric) -> Path:
    """``problems.csv`` shaped like the BioMysteryBench card's columns."""
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "problems.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "question",
                biomysterybench.ORACLE_FIELD,
                "allowed_domains",
                "human_solvable",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "syn1",
                "question": "Synthetic?",
                biomysterybench.ORACLE_FIELD: rubric,
                "allowed_domains": "",
                "human_solvable": "True",
            }
        )
    biomysterybench.standardize(
        raw_dir=raw, for_solver_dir=tmp_path / "for_solver", eval_dir=tmp_path / "eval"
    )
    return tmp_path / "eval"


def _payloads(eval_dir: Path) -> list[dict]:
    lines = (eval_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["answer"] for line in lines if line.strip()]


def _run_evaluate(eval_dir: Path, tmp_path: Path, submission: list[dict]):
    sub = tmp_path / "submission.json"
    sub.write_text(json.dumps(submission), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(eval_dir / "evaluate.py"),
            "--submission",
            str(sub),
            "--answers",
            str(eval_dir / "answers.jsonl"),
        ],
        capture_output=True,
        text=True,
    )


def _oracle_error(benchmark: str, answers: Path) -> OracleIntegrityError:
    """Run the default (strict) scorer and return the refusal it raises."""
    try:
        score_submission(benchmark, [], answers=answers)
    except OracleIntegrityError as exc:
        return exc
    raise AssertionError("scorer did not refuse the oracle")


def _write_answers(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Per-adapter contract: the scorer's reference field exists in what the
# adapter writes from a schema-shaped upstream record.
# ---------------------------------------------------------------------------


class TestReferenceFieldContract:
    def test_every_adapter_declares_a_reference_field(self):
        # Arrange
        expected = {"corebench": "value", "bixbench": "ideal", "biomysterybench": "rubric"}
        # Act
        fields = reference_fields()
        # Assert
        assert fields == expected

    def test_corebench_payload_carries_reference_field(self):
        # Arrange — core_train.json shape: results = list of {question: value}.
        rec = {
            "capsule_id": "capsule-0000001",
            "task_prompt": "Run it.",
            "results": [{"Report the synthetic metric.": 1.5}],
        }
        # Act
        _tasks, answers = corebench._split_record(rec)
        # Assert
        field = reference_fields()["corebench"]
        assert answers and all(a["answer"].get(field) is not None for a in answers)

    def test_bixbench_payload_carries_reference_field(self, tmp_path):
        # Arrange
        records = [_bixbench_record("syn-1-q1", "synthetic")]
        # Act
        eval_dir = _standardize_bixbench(tmp_path, records)
        # Assert
        field = reference_fields()["bixbench"]
        assert all(
            isinstance(p.get(field), str) and p[field] for p in _payloads(eval_dir)
        )

    def test_biomysterybench_payload_carries_reference_field(self, tmp_path):
        # Arrange
        rubric = "Award marks for synthetic X"
        # Act
        eval_dir = _standardize_biomysterybench(tmp_path, rubric)
        # Assert
        field = reference_fields()["biomysterybench"]
        assert all(
            isinstance(p.get(field), str) and p[field] for p in _payloads(eval_dir)
        )


# ---------------------------------------------------------------------------
# BixBench grading (#115)
# ---------------------------------------------------------------------------


class TestBixbenchScoring:
    def test_issue_repro_numeric_ideal_with_true_flag_is_correct(self, tmp_path):
        # Arrange — the issue's dataset-blind repro.
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "bixbench/x-q1", "answer": {"answer": True, "ideal": "0.5"}}],
        )
        # Act
        row = score_submission(
            "bixbench", [{"task_id": "bixbench/x-q1", "answer": 0.5}], answers=ans
        )[0]
        # Assert
        assert (row["verdict"], row["expected"]) == ("correct", "0.5")

    @pytest.mark.parametrize("flag", [True, False])
    def test_boolean_flag_never_grades(self, tmp_path, flag):
        # Arrange — submitting the flag itself must not match a text reference.
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "bixbench/x-q1",
                    "answer": {"answer": flag, "ideal": "gene X"},
                }
            ],
        )
        # Act
        row = score_submission(
            "bixbench", [{"task_id": "bixbench/x-q1", "answer": str(flag)}], answers=ans
        )[0]
        # Assert
        assert row["verdict"] == "wrong"

    @pytest.mark.parametrize(
        "submitted,verdict",
        [(1.52, "correct"), (1.5, "correct"), (1.6, "wrong"), ("1.53", "correct")],
    )
    def test_range_ideal_scores_as_interval(self, tmp_path, submitted, verdict):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "bixbench/x-q1",
                    "answer": {"answer": False, "ideal": "(1.50,1.54)"},
                }
            ],
        )
        # Act
        row = score_submission(
            "bixbench", [{"task_id": "bixbench/x-q1", "answer": submitted}], answers=ans
        )[0]
        # Assert
        assert row["verdict"] == verdict

    def test_text_ideal_scores_by_string_policy(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "bixbench/x-q1",
                    "answer": {"answer": True, "ideal": "Gene X"},
                }
            ],
        )
        # Act
        row = score_submission(
            "bixbench",
            [{"task_id": "bixbench/x-q1", "answer": " gene x "}],
            answers=ans,
        )[0]
        # Assert
        assert row["verdict"] == "correct"

    def test_missing_ideal_raises_payload_free_diagnostic(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "bixbench/x-q1", "answer": {"answer": True}}],
        )
        # Act
        error = _oracle_error("bixbench", ans)
        # Assert — names the kind and the field, never a reference value.
        assert error.diagnostics == [
            {"line": 1, "kind": "missing_reference_field", "field": "ideal"}
        ]

    def test_missing_ideal_error_message_names_kind_and_field(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "bixbench/x-q1", "answer": {"answer": True}}],
        )
        # Act
        message = str(_oracle_error("bixbench", ans))
        # Assert
        assert "missing_reference_field: 1" in message and "reference field: ideal" in message

    @pytest.mark.parametrize("ideal", [None, "", "  "])
    def test_empty_ideal_is_invalid_reference_not_wrong(self, tmp_path, ideal):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "bixbench/x-q1", "answer": {"answer": True, "ideal": ideal}}],
        )
        # Act
        row = score_submission(
            "bixbench", [{"task_id": "bixbench/x-q1", "answer": "a"}], answers=ans
        )[0]
        # Assert
        assert (row["verdict"], row["reference_error_kind"]) == (
            "invalid_reference",
            "empty_reference",
        )

    @pytest.mark.parametrize("ideal", [_ABSENT, None, "", "   ", 3])
    def test_standardize_refuses_record_without_text_ideal(self, tmp_path, ideal):
        # Arrange
        raw = _stage_bixbench(tmp_path, [_bixbench_record("syn-1-q1", ideal)])
        # Act
        # Assert
        with pytest.raises(ValueError, match="'ideal'"):
            bixbench.standardize(
                raw_dir=raw,
                for_solver_dir=tmp_path / "for_solver",
                eval_dir=tmp_path / "eval",
            )


class TestBixbenchEndToEnd:
    @pytest.fixture
    def eval_dir(self, tmp_path):
        return _standardize_bixbench(
            tmp_path,
            [
                _bixbench_record("syn-1-q1", "0.5", answer=True),
                _bixbench_record(
                    "syn-1-q2", "(1.50,1.54)", answer=False, eval_mode="range_verifier"
                ),
                _bixbench_record("syn-1-q3", "gene X", answer=True),
            ],
        )

    _MATCHING = [
        {"task_id": "bixbench/syn-1-q1", "answer": 0.5},
        {"task_id": "bixbench/syn-1-q2", "answer": 1.52},
        {"task_id": "bixbench/syn-1-q3", "answer": "gene x"},
    ]

    def test_score_submission_grades_matching_answers_correct(self, eval_dir):
        # Arrange
        submission = self._MATCHING
        # Act
        rows = score_submission("bixbench", submission, answers=eval_dir)
        # Assert
        assert [r["verdict"] for r in rows] == ["correct"] * 3

    def test_generated_evaluate_grades_matching_answers_correct(
        self, eval_dir, tmp_path
    ):
        # Arrange
        submission = self._MATCHING
        # Act
        out = _run_evaluate(eval_dir, tmp_path, submission)
        # Assert
        assert json.loads(out.stdout)["score"] == 1.0

    def test_generated_evaluate_refuses_missing_ideal(self, eval_dir, tmp_path):
        # Arrange — strip the reference key from the written oracle.
        path = eval_dir / "answers.jsonl"
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        for rec in recs:
            rec["answer"].pop("ideal")
        _write_answers(path, recs)
        # Act
        out = _run_evaluate(eval_dir, tmp_path, self._MATCHING)
        # Assert
        assert out.returncode != 0 and "'ideal'" in out.stderr


class TestOtherAdapters:
    def test_biomysterybench_refuses_empty_rubric(self, tmp_path):
        # Arrange
        rubric = ""
        # Act
        # Assert
        with pytest.raises(ValueError, match="answer_rubric"):
            _standardize_biomysterybench(tmp_path, rubric)

    def test_biomysterybench_scores_needs_rubric(self, tmp_path):
        # Arrange
        eval_dir = _standardize_biomysterybench(tmp_path, "Award marks for synthetic X")
        # Act
        row = score_submission(
            "biomysterybench",
            [{"task_id": "biomysterybench/syn1", "answer": "X"}],
            answers=eval_dir,
        )[0]
        # Assert
        assert row["verdict"] == "needs_rubric"

    def test_corebench_payload_missing_value_fails_loudly(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "corebench/capsule-1__question_x", "answer": {"other": 1}}],
        )
        # Act
        # Assert
        with pytest.raises(OracleIntegrityError, match="missing_reference_field"):
            score_submission("corebench", [], answers=ans)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("(1.5,1.6)", (1.5, 1.6)),
        ("[ -2e-3 , 4 ]", (-0.002, 4.0)),
        ("(2,1)", (1.0, 2.0)),
        ("1.5", None),
        ("gene X", None),
        (None, None),
    ],
)
def test_parse_interval_reads_only_bracketed_numeric_ranges(text, expected):
    # Arrange
    value = text
    # Act
    parsed = parse_interval(value)
    # Assert
    assert parsed == expected
