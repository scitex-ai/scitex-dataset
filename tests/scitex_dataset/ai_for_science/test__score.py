#!/usr/bin/env python3
"""Tests for ai_for_science._score (correctness scoring vs the oracle).

No mocks / monkeypatch: real answers.jsonl fixtures on disk, real
submission objects, and real comparator calls. Covers every eval family
(numeric multi-ref PI, numeric n==1 sig-fig, string, set-equality incl.
the unhashable fallback) and every verdict
(correct/wrong/abstain/malformed×kind/needs_rubric).
"""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _score
from scitex_dataset.ai_for_science._score import (
    score_numeric,
    score_set,
    score_string,
    score_submission,
)


def _write_answers(path: Path, records: list[dict]) -> Path:
    """Write ``records`` as an answers.jsonl and return the file path."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _one_numeric_answers(tmp_path, value=0.9996):
    return _write_answers(
        tmp_path / "answers.jsonl",
        [{"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": value}}],
    )


# ---------------------------------------------------------------------------
# Family comparators (called directly)
# ---------------------------------------------------------------------------


class TestNumericFamily:
    def test_multi_ref_pi_accepts_in_range(self):
        # Arrange — three reference reruns; a value inside the 95% PI.
        values = [4.20, 4.21, 4.19]
        # Act
        ok = score_numeric(values, 4.20)
        # Assert
        assert ok is True

    def test_multi_ref_pi_rejects_out_of_range(self):
        # Arrange
        values = [4.20, 4.21, 4.19]
        # Act
        ok = score_numeric(values, 9.99)
        # Assert
        assert ok is False

    def test_n1_sigfig_accepts_at_published_precision(self):
        # Arrange — n==1 falls back to sig-fig tolerance (0.99956 ≈ 0.9996).
        values = [0.9996]
        # Act
        ok = score_numeric(values, 0.99956)
        # Assert
        assert ok is True

    def test_n1_sigfig_rejects_precision_loss(self):
        # Arrange — 0.59 is a real precision loss vs 0.59375 (5 SF).
        values = [0.59375]
        # Act
        ok = score_numeric(values, 0.59)
        # Assert
        assert ok is False

    def test_degenerate_pi_uses_sigfig(self):
        # Arrange — identical reruns collapse the PI to zero width.
        values = [0.9996, 0.9996, 0.9996]
        # Act
        ok = score_numeric(values, 0.99956)
        # Assert
        assert ok is True


class TestStringFamily:
    def test_case_and_whitespace_insensitive_match(self):
        # Arrange
        expected = "Positive"
        # Act
        ok = score_string(expected, "  positive ")
        # Assert
        assert ok is True

    def test_mismatch_is_false(self):
        # Arrange
        expected = "positive"
        # Act
        ok = score_string(expected, "negative")
        # Assert
        assert ok is False


class TestSetFamily:
    def test_order_insensitive_match(self):
        # Arrange
        expected = ["b", "a", "c"]
        # Act
        ok = score_set(expected, ["c", "b", "a"])
        # Assert
        assert ok is True

    def test_non_list_reported_is_false(self):
        # Arrange
        expected = ["a", "b"]
        # Act
        ok = score_set(expected, "a,b")
        # Assert
        assert ok is False

    def test_unhashable_fallback_returns_false(self):
        # Arrange — a value whose str() raises, forcing the ordered ==
        # fallback path (real object, no mock).
        class _BadStr:
            def __str__(self):
                raise TypeError("cannot stringify")

        expected = [_BadStr()]
        # Act
        ok = score_set(expected, [1])
        # Assert
        assert ok is False


# ---------------------------------------------------------------------------
# score_submission — verdicts end to end
# ---------------------------------------------------------------------------


class TestVerdictCorrectWrong:
    def test_numeric_within_sigfig_is_correct(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "correct"

    def test_numeric_out_of_range_is_wrong(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.5}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "wrong"

    def test_string_match_is_correct(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": "Yes"}}],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": "yes"}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "correct"

    def test_set_equality_match_is_correct(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "corebench/capsule-1__hard__q0",
                    "answer": {"value": ["a", "b"]},
                }
            ],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": ["b", "a"]}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "correct"


class TestVerdictAbstain:
    def test_null_with_abstain_reason_is_abstain(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": None,
                "reason": "agent abstained: insufficient evidence",
            }
        ]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "abstain"

    def test_canonical_text_with_abstain_reason_is_abstain(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": "cannot determine from available evidence",
                "reason": "agent abstained: unclear",
            }
        ]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "abstain"


class TestVerdictMalformed:
    def test_no_file_is_no_submission(self, tmp_path):
        # Arrange — a submission path that does not exist.
        ans = _one_numeric_answers(tmp_path)
        missing = tmp_path / "nope.json"
        # Act
        recs = score_submission("corebench", missing, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "no_submission"

    def test_missing_task_is_no_submission(self, tmp_path):
        # Arrange — a valid submission that omits the oracle's task.
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-9__hard__q0", "answer": 1.0}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "no_submission"

    def test_non_json_is_unparseable(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        # Act
        recs = score_submission("corebench", bad, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "unparseable"

    def test_empty_array_is_empty(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = []
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "empty"

    def test_bare_null_without_reason_is_empty(self, tmp_path):
        # Arrange — null answer, no abstention reason.
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": None}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "empty"

    def test_schema_invalid_is_schema_invalid(self, tmp_path):
        # Arrange — a non-empty submission that fails structural validation.
        ans = _one_numeric_answers(tmp_path)
        sub = [{"answer": 1.0}]  # missing task_id
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "schema_invalid"


class TestVerdictNeedsRubric:
    def test_rubric_answer_is_needs_rubric(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "biomysterybench/x", "answer": {"rubric": "Award 1 pt if…"}}],
        )
        sub = [{"task_id": "biomysterybench/x", "answer": "my analysis"}]
        # Act
        recs = score_submission("biomysterybench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "needs_rubric"

    def test_rubric_record_carries_hint(self, tmp_path):
        # Arrange
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "biomysterybench/x", "answer": {"rubric": "Award 1 pt if…"}}],
        )
        sub = [{"task_id": "biomysterybench/x", "answer": "my analysis"}]
        # Act
        recs = score_submission("biomysterybench", sub, answers=ans)
        # Assert
        assert "hint" in recs[0]


# ---------------------------------------------------------------------------
# Record shape + join-key alignment
# ---------------------------------------------------------------------------


class TestRecordShape:
    def test_record_carries_submitted(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["submitted"] == 0.99956

    def test_record_carries_expected(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["expected"] == 0.9996


class TestJoinKeyAlignment:
    def test_repeated_task_id_groups_into_multi_ref_pi(self, tmp_path):
        # Arrange — the same task_id on three oracle lines = 3 reference
        # values, so the numeric prediction interval (not sig-fig) fires.
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": 4.20}},
                {"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": 4.21}},
                {"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": 4.19}},
            ],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 4.20}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert — one joined record, scored correct within the PI.
        assert len(recs) == 1 and recs[0]["verdict"] == "correct"

    def test_eval_dir_is_accepted_as_answers_arg(self, tmp_path):
        # Arrange — pass the eval DIR rather than the answers.jsonl file.
        _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
        # Act
        recs = score_submission("corebench", sub, answers=tmp_path)
        # Assert
        assert recs[0]["verdict"] == "correct"


class TestNativeFromTaskId:
    def test_strips_prefix_and_suffix(self):
        # Arrange
        task_id = "corebench/capsule-7038571__hard__q0"
        # Act
        native = _score._native_from_task_id(task_id)
        # Assert
        assert native == "capsule-7038571"

    def test_non_string_returns_none(self):
        # Arrange
        task_id = 123
        # Act
        native = _score._native_from_task_id(task_id)
        # Assert
        assert native is None


class TestInternalHelpers:
    def test_t_crit_nonpositive_df_is_nan(self):
        # Arrange
        import math

        df = 0
        # Act
        crit = _score._t_crit_95(df)
        # Assert
        assert math.isnan(crit)

    def test_to_float_bool_is_none(self):
        # Arrange
        value = True
        # Act
        out = _score._to_float(value)
        # Assert
        assert out is None

    def test_to_float_numeric_string(self):
        # Arrange
        value = "3.5"
        # Act
        out = _score._to_float(value)
        # Assert
        assert out == 3.5

    def test_to_float_non_numeric_string_is_none(self):
        # Arrange
        value = "abc"
        # Act
        out = _score._to_float(value)
        # Assert
        assert out is None

    def test_to_float_other_type_is_none(self):
        # Arrange
        value = [1]
        # Act
        out = _score._to_float(value)
        # Assert
        assert out is None

    def test_score_numeric_empty_values_is_false(self):
        # Arrange
        values = []
        # Act
        ok = score_numeric(values, 1.0)
        # Assert
        assert ok is False

    def test_expected_value_non_dict_passthrough(self):
        # Arrange
        payload = "foo"
        # Act
        out = _score._expected_value(payload)
        # Assert
        assert out == "foo"

    def test_is_abstention_reason_but_real_answer_is_false(self):
        # Arrange — reason present, but a concrete answer is not abstention.
        submitted = "42"
        reason = "agent abstained: unclear"
        # Act
        out = _score._is_abstention(submitted, reason)
        # Assert
        assert out is False


class TestLoadOracleEdges:
    def test_blank_and_bad_lines_are_skipped(self, tmp_path):
        # Arrange — blank line, malformed JSON, a non-dict line, and a
        # line whose task_id is not a string are all ignored; the one
        # good record survives.
        p = tmp_path / "answers.jsonl"
        p.write_text(
            "\n"
            "{not json\n"
            + json.dumps([1, 2]) + "\n"
            + json.dumps({"task_id": 5, "answer": {"value": 1}}) + "\n"
            + json.dumps(
                {"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": 1}}
            )
            + "\n"
        )
        # Act
        grouped = _score._load_oracle(p)
        # Assert
        assert list(grouped) == ["corebench/capsule-1__hard__q0"]


class TestWholeSubmissionAndFamilies:
    def test_none_submission_is_no_submission(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        # Act
        recs = score_submission("corebench", None, answers=ans)
        # Assert
        assert recs[0]["malformed_kind"] == "no_submission"

    def test_answers_none_resolves_via_dataset_root(self, tmp_path):
        # Arrange — stage the eval dir under the resolved dataset layout.
        eval_dir = tmp_path / "ai-for-science" / "corebench" / "eval"
        eval_dir.mkdir(parents=True)
        _write_answers(
            eval_dir / "answers.jsonl",
            [{"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": 0.9996}}],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
        # Act
        recs = score_submission("corebench", sub, answers=None, dataset_root=tmp_path)
        # Assert
        assert recs[0]["verdict"] == "correct"

    def test_bool_expected_uses_string_family(self, tmp_path):
        # Arrange — a boolean oracle value routes through the string family.
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [{"task_id": "corebench/capsule-1__hard__q0", "answer": {"value": True}}],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": True}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "correct"

    def test_non_numeric_answer_on_numeric_task_is_malformed(self, tmp_path):
        # Arrange
        ans = _one_numeric_answers(tmp_path)
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": "not a number"}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "malformed"

    def test_unsupported_expected_type_is_malformed(self, tmp_path):
        # Arrange — a nested-dict oracle value matches no eval family.
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "corebench/capsule-1__hard__q0",
                    "answer": {"value": {"nested": 1}},
                }
            ],
        )
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": "x"}]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        assert recs[0]["verdict"] == "malformed"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
