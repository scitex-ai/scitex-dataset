#!/usr/bin/env python3
"""Tests for ai_for_science._score (correctness scoring vs the oracle).

No mocks / monkeypatch: real answers.jsonl fixtures on disk, real
submission objects, and real comparator calls. Covers every eval family
(numeric multi-ref PI, numeric n==1 sig-fig, string, set-equality incl.
the unhashable fallback) and every verdict
(correct/wrong/abstain/malformed×kind/needs_rubric).
"""

import json
from types import SimpleNamespace
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

    @pytest.fixture
    def _case_missing_task_is_no_submission(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        ans = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': 'corebench/capsule-1__hard__q0', 'answer': {'value': 0.9996}}, {'task_id': 'corebench/capsule-9__hard__q0', 'answer': {'value': 1.0}}])
        sub = [{'task_id': 'corebench/capsule-9__hard__q0', 'answer': 1.0}]

        def run_case(*, capture_error=False):
            recs = score_submission('corebench', sub, answers=ans)
            return (recs,)
        return run_case

    def test_missing_task_is_no_submission(self, _case_missing_task_is_no_submission):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_missing_task_is_no_submission
        # Act
        recs, = run_case()
        # Assert
        assert recs[0]['malformed_kind'] == 'no_submission'

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

    def test_reasonless_null_does_not_invalidate_sibling_row(self, tmp_path):
        # Arrange — a 2-task oracle; q0 answered correctly, q1 a reasonless
        # null. The reasonless null is a per-row 'empty', NOT a global
        # schema_invalid — so q0 must still score on its own. (Guards the
        # scorer against the validator's new missing_reason hard error.)
        ans = _write_answers(
            tmp_path / "answers.jsonl",
            [
                {
                    "task_id": "corebench/capsule-1__hard__q0",
                    "answer": {"value": 0.9996},
                },
                {"task_id": "corebench/capsule-1__hard__q1", "answer": {"value": 0.5}},
            ],
        )
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956},
            {"task_id": "corebench/capsule-1__hard__q1", "answer": None},
        ]
        # Act
        recs = score_submission("corebench", sub, answers=ans)
        # Assert
        q0 = next(r for r in recs if r["task_id"] == "corebench/capsule-1__hard__q0")
        assert q0["verdict"] == "correct"

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
            + json.dumps([1, 2])
            + "\n"
            + json.dumps({"task_id": 5, "answer": {"value": 1}})
            + "\n"
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

    @pytest.fixture
    def _case_unsupported_expected_type_is_invalid_reference(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        ans = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': 'corebench/capsule-1__hard__q0', 'answer': {'value': {'nested': 1}}}])
        sub = [{'task_id': 'corebench/capsule-1__hard__q0', 'answer': 'x'}]

        def run_case(*, capture_error=False):
            recs = score_submission('corebench', sub, answers=ans)
            return (recs,)
        return run_case

    def test_unsupported_expected_type_is_invalid_reference_invalid_verdict(self, _case_unsupported_expected_type_is_invalid_reference):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_unsupported_expected_type_is_invalid_reference
        # Act
        recs, = run_case()
        # Assert
        assert recs[0]['verdict'] == 'invalid_reference'

    def test_unsupported_expected_type_is_invalid_reference_unsupported_type_diagnostic(self, _case_unsupported_expected_type_is_invalid_reference):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_unsupported_expected_type_is_invalid_reference
        # Act
        recs, = run_case()
        # Assert
        assert recs[0]['reference_error_kind'] == 'unsupported_reference_type'


class TestReferenceIntegrity:
    @pytest.fixture
    def _case_strict_default_refuses_missing_answer_field(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid}])

        def run_case(*, capture_error=False):
            error = None
            if capture_error:
                try:
                    score_submission('corebench', [{'task_id': tid, 'answer': 5}], answers=path)
                except _score.OracleIntegrityError as captured:
                    error = SimpleNamespace(value=captured)
            else:
                score_submission('corebench', [{'task_id': tid, 'answer': 5}], answers=path)
            return (error,)
        return run_case

    def test_strict_default_refuses_missing_answer_field_raises_integrity_error(self, _case_strict_default_refuses_missing_answer_field):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_strict_default_refuses_missing_answer_field
        # Act
        ctx = pytest.raises(_score.OracleIntegrityError)
        # Assert
        with ctx:
            run_case()

    def test_strict_default_refuses_missing_answer_field_missing_field_diagnostic(self, _case_strict_default_refuses_missing_answer_field):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_strict_default_refuses_missing_answer_field
        # Act
        error, = run_case(capture_error=True)
        # Assert
        assert error.value.diagnostics == [{'line': 1, 'kind': 'missing_answer_field'}]

    @pytest.fixture
    def _case_present_null_reference_is_not_missing_answer_field(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid, 'answer': None}])

        def run_case(*, capture_error=False):
            row = score_submission('corebench', [{'task_id': tid, 'answer': 5}], answers=path)[0]
            return (row,)
        return run_case

    def test_present_null_reference_is_not_missing_answer_field_invalid_reference_verdict(self, _case_present_null_reference_is_not_missing_answer_field):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_reference_is_not_missing_answer_field
        # Act
        row, = run_case()
        # Assert
        assert row['verdict'] == 'invalid_reference'

    def test_present_null_reference_is_not_missing_answer_field_no_missing_field_diagnostic(self, _case_present_null_reference_is_not_missing_answer_field):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_present_null_reference_is_not_missing_answer_field
        # Act
        row, = run_case()
        # Assert
        assert 'oracle_diagnostics' not in row

    @pytest.fixture
    def _case_tolerant_missing_field_has_payload_free_diagnostics(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid}, {'task_id': tid, 'answer': {'value': 5}}])

        def run_case(*, capture_error=False):
            rows = score_submission('corebench', [{'task_id': tid, 'answer': 5}], answers=path, strict_oracle=False)
            return (rows,)
        return run_case

    def test_tolerant_missing_field_has_payload_free_diagnostics_valid_peer_verdict(self, _case_tolerant_missing_field_has_payload_free_diagnostics):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_missing_field_has_payload_free_diagnostics
        # Act
        rows, = run_case()
        # Assert
        assert rows[0]['verdict'] == 'correct'

    def test_tolerant_missing_field_has_payload_free_diagnostics_invalid_integrity_flag(self, _case_tolerant_missing_field_has_payload_free_diagnostics):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_missing_field_has_payload_free_diagnostics
        # Act
        rows, = run_case()
        # Assert
        assert rows[0]['oracle_integrity'] == 'invalid'

    def test_tolerant_missing_field_has_payload_free_diagnostics_missing_field_diagnostic(self, _case_tolerant_missing_field_has_payload_free_diagnostics):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_missing_field_has_payload_free_diagnostics
        # Act
        rows, = run_case()
        # Assert
        assert rows[0]['oracle_diagnostics'] == [{'line': 1, 'kind': 'missing_answer_field'}]

    @pytest.fixture
    def _case_public_default_refuses_malformed_oracle_records(self, tmp_path, has_valid):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        path = tmp_path / 'answers.jsonl'
        tid = 'corebench/capsule-1__hard__q0'
        valid = json.dumps({'task_id': tid, 'answer': {'value': 0.9996}}) + '\n'
        path.write_text((valid if has_valid else '') + 'not json\n[]\n')

        def run_case(*, capture_error=False):
            error = None
            if capture_error:
                try:
                    score_submission('corebench', [{'task_id': tid, 'answer': 0.9996}], answers=path)
                except _score.OracleIntegrityError as captured:
                    error = SimpleNamespace(value=captured)
            else:
                score_submission('corebench', [{'task_id': tid, 'answer': 0.9996}], answers=path)
            return (error,)
        return run_case

    @pytest.mark.parametrize('has_valid', [False, True])
    def test_public_default_refuses_malformed_oracle_records_raises_integrity_error(self, _case_public_default_refuses_malformed_oracle_records, has_valid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_public_default_refuses_malformed_oracle_records
        # Act
        ctx = pytest.raises(_score.OracleIntegrityError)
        # Assert
        with ctx:
            run_case()

    @pytest.mark.parametrize('has_valid', [False, True])
    def test_public_default_refuses_malformed_oracle_records_two_record_diagnostics(self, _case_public_default_refuses_malformed_oracle_records, has_valid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_public_default_refuses_malformed_oracle_records
        # Act
        error, = run_case(capture_error=True)
        # Assert
        assert len(error.value.diagnostics) == 2

    @pytest.mark.parametrize('has_valid', [False, True])
    def test_public_default_refuses_malformed_oracle_records_private_payload_absent(self, _case_public_default_refuses_malformed_oracle_records, has_valid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_public_default_refuses_malformed_oracle_records
        # Act
        error, = run_case(capture_error=True)
        # Assert
        assert '0.9996' not in str(error.value)

    @pytest.fixture
    def malformed_oracle_path(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        return path

    @pytest.mark.parametrize("has_valid", [False, True])
    def test_explicit_tolerant_mode_reports_both_invalid_records(self, malformed_oracle_path, has_valid):
        # Arrange
        path = malformed_oracle_path
        tid = "corebench/capsule-1__hard__q0"
        valid = json.dumps({"task_id": tid, "answer": {"value": 0.9996}}) + "\n"
        path.write_text((valid if has_valid else "") + "not json\n[]\n")
        diagnostics = []
        # Act
        _score._load_oracle(path, diagnostics=diagnostics)
        # Assert
        assert len(diagnostics) == 2

    @pytest.fixture
    def _case_tolerant_mode_retains_the_valid_peer(self, malformed_oracle_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        path = malformed_oracle_path
        tid = 'corebench/capsule-1__hard__q0'
        valid = json.dumps({'task_id': tid, 'answer': {'value': 0.9996}}) + '\n'
        path.write_text(valid + 'not json\n[]\n')
        diagnostics = []
        _score._load_oracle(path, diagnostics=diagnostics)

        def run_case(*, capture_error=False):
            rows = score_submission('corebench', [{'task_id': tid, 'answer': 0.9996}], answers=path, strict_oracle=False)
            return (diagnostics, rows)
        return run_case

    def test_tolerant_mode_retains_the_valid_peer_valid_peer_verdict(self, _case_tolerant_mode_retains_the_valid_peer):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_mode_retains_the_valid_peer
        # Act
        diagnostics, rows = run_case()
        # Assert
        assert len(rows) == 1 and rows[0]['verdict'] == 'correct'

    def test_tolerant_mode_retains_the_valid_peer_invalid_integrity_flag(self, _case_tolerant_mode_retains_the_valid_peer):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_mode_retains_the_valid_peer
        # Act
        diagnostics, rows = run_case()
        # Assert
        assert rows[0]['oracle_integrity'] == 'invalid'

    def test_tolerant_mode_retains_the_valid_peer_record_diagnostics_retained(self, _case_tolerant_mode_retains_the_valid_peer):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_tolerant_mode_retains_the_valid_peer
        # Act
        diagnostics, rows = run_case()
        # Assert
        assert rows[0]['oracle_diagnostics'] == diagnostics

    def test_tolerant_mode_without_valid_records_returns_no_score(self, malformed_oracle_path):
        # Arrange
        path = malformed_oracle_path
        tid = "corebench/capsule-1__hard__q0"
        path.write_text("not json\n[]\n")
        # Act
        rows = score_submission("corebench", [{"task_id": tid, "answer": 0.9996}], answers=path, strict_oracle=False)
        # Assert
        assert rows == []

    @pytest.fixture
    def appended_invalid_oracle_path(self, tmp_path):
        path = _one_numeric_answers(tmp_path)
        with path.open("a") as fh:
            fh.write("not json\n[]\n")
        yield path

    def test_tolerant_oracle_retains_valid_group(self, appended_invalid_oracle_path):
        # Arrange
        path = appended_invalid_oracle_path
        diagnostics = []
        # Act
        oracle = _score._load_oracle(path, diagnostics=diagnostics)
        # Assert
        assert len(oracle) == 1

    def test_tolerant_oracle_attributes_invalid_record_diagnostics(self, appended_invalid_oracle_path):
        # Arrange
        path = appended_invalid_oracle_path
        diagnostics = []
        # Act
        _score._load_oracle(path, diagnostics=diagnostics)
        # Assert
        assert diagnostics == [{"line": 2, "kind": "unparseable_record"},
                               {"line": 3, "kind": "record_not_object"}]

    def test_strict_oracle_refuses_appended_invalid_records(self, appended_invalid_oracle_path):
        # Arrange
        path = appended_invalid_oracle_path
        # Act
        ctx = pytest.raises(_score.OracleIntegrityError)
        # Assert
        with ctx:
            score_submission("corebench", [], answers=path, strict_oracle=True)

    @pytest.fixture
    def _case_strict_oracle_retains_diagnostics_from_tolerant_reader(self, appended_invalid_oracle_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        path = appended_invalid_oracle_path
        diagnostics = []
        _score._load_oracle(path, diagnostics=diagnostics)

        def run_case(*, capture_error=False):
            caught = None
            if capture_error:
                try:
                    score_submission('corebench', [], answers=path, strict_oracle=True)
                except _score.OracleIntegrityError as captured:
                    caught = SimpleNamespace(value=captured)
            else:
                score_submission('corebench', [], answers=path, strict_oracle=True)
            return (caught, diagnostics)
        return run_case

    def test_strict_oracle_retains_diagnostics_from_tolerant_reader_raises_integrity_error(self, _case_strict_oracle_retains_diagnostics_from_tolerant_reader):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_strict_oracle_retains_diagnostics_from_tolerant_reader
        # Act
        ctx = pytest.raises(_score.OracleIntegrityError)
        # Assert
        with ctx:
            run_case()

    def test_strict_oracle_retains_diagnostics_from_tolerant_reader_record_diagnostics_retained(self, _case_strict_oracle_retains_diagnostics_from_tolerant_reader):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_strict_oracle_retains_diagnostics_from_tolerant_reader
        # Act
        caught, diagnostics = run_case(capture_error=True)
        # Assert
        assert caught.value.diagnostics == diagnostics

    def test_strict_oracle_retains_diagnostics_from_tolerant_reader_private_payload_absent(self, _case_strict_oracle_retains_diagnostics_from_tolerant_reader):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_strict_oracle_retains_diagnostics_from_tolerant_reader
        # Act
        caught, diagnostics = run_case(capture_error=True)
        # Assert
        assert '0.9996' not in str(caught.value)

    @pytest.fixture
    def _case_invalid_numeric_reference_is_not_dropped(self, tmp_path, invalid):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid, 'answer': {'value': value}} for value in (0.9996, invalid, 0.9996)])

        def run_case(*, capture_error=False):
            row = score_submission('corebench', [{'task_id': tid, 'answer': 0.9996}], answers=path)[0]
            return (row,)
        return run_case

    @pytest.mark.parametrize('invalid', [None, True, 'not numeric', 'nan', float('inf')])
    def test_invalid_numeric_reference_is_not_dropped_invalid_reference_verdict(self, _case_invalid_numeric_reference_is_not_dropped, invalid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_reference_is_not_dropped
        # Act
        row, = run_case()
        # Assert
        assert row['verdict'] == 'invalid_reference'

    @pytest.mark.parametrize('invalid', [None, True, 'not numeric', 'nan', float('inf')])
    def test_invalid_numeric_reference_is_not_dropped_all_sample_count(self, _case_invalid_numeric_reference_is_not_dropped, invalid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_reference_is_not_dropped
        # Act
        row, = run_case()
        # Assert
        assert row['n_references'] == 3

    @pytest.mark.parametrize('invalid', [None, True, 'not numeric', 'nan', float('inf')])
    def test_invalid_numeric_reference_is_not_dropped_numeric_reference_diagnostic(self, _case_invalid_numeric_reference_is_not_dropped, invalid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_reference_is_not_dropped
        # Act
        row, = run_case()
        # Assert
        assert row['reference_error_kind'] == 'invalid_numeric_reference'

    @pytest.fixture
    def _case_invalid_numeric_submission_is_malformed(self, tmp_path, invalid):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _one_numeric_answers(tmp_path)

        def run_case(*, capture_error=False):
            row = score_submission('corebench', [{'task_id': tid, 'answer': invalid}], answers=path)[0]
            return (row,)
        return run_case

    @pytest.mark.parametrize('invalid', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_is_malformed_malformed_verdict(self, _case_invalid_numeric_submission_is_malformed, invalid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_is_malformed
        # Act
        row, = run_case()
        # Assert
        assert row['verdict'] == 'malformed'

    @pytest.mark.parametrize('invalid', [True, 'nan', 'inf', 10 ** 400])
    def test_invalid_numeric_submission_is_malformed_numeric_answer_diagnostic(self, _case_invalid_numeric_submission_is_malformed, invalid):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_invalid_numeric_submission_is_malformed
        # Act
        row, = run_case()
        # Assert
        assert row['malformed_kind'] == 'invalid_numeric_answer'

    @pytest.fixture
    def _case_arithmetic_reference_failure_preserves_valid_peer(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        bad = 'corebench/capsule-1__hard__q0'
        good = 'corebench/capsule-1__hard__q1'
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': bad, 'answer': {'value': 1e+308}}, {'task_id': bad, 'answer': {'value': -1e+308}}, {'task_id': good, 'answer': {'value': 0.9996}}])

        def run_case(*, capture_error=False):
            rows = score_submission('corebench', [{'task_id': bad, 'answer': 0}, {'task_id': good, 'answer': 0.9996}], answers=path)
            return (rows,)
        return run_case

    def test_arithmetic_reference_failure_preserves_valid_peer_both_task_verdicts(self, _case_arithmetic_reference_failure_preserves_valid_peer):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_arithmetic_reference_failure_preserves_valid_peer
        # Act
        rows, = run_case()
        # Assert
        assert [r['verdict'] for r in rows] == ['invalid_reference', 'correct']

    def test_arithmetic_reference_failure_preserves_valid_peer_arithmetic_reference_diagnostic(self, _case_arithmetic_reference_failure_preserves_valid_peer):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_arithmetic_reference_failure_preserves_valid_peer
        # Act
        rows, = run_case()
        # Assert
        assert rows[0]['reference_error_kind'] == 'numeric_reference_arithmetic'


    @pytest.fixture
    def _case_partial_submission_keeps_missing_assigned_question(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tids = ['corebench/capsule-1__hard__q0', 'corebench/capsule-1__hard__q1']
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid, 'answer': {'value': 0.9996}} for tid in tids])

        def run_case(*, capture_error=False):
            rows = score_submission('corebench', [{'task_id': tids[0], 'answer': 0.9996}], answers=path)
            return (rows,)
        return run_case

    def test_partial_submission_keeps_missing_assigned_question_both_task_verdicts(self, _case_partial_submission_keeps_missing_assigned_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_partial_submission_keeps_missing_assigned_question
        # Act
        rows, = run_case()
        # Assert
        assert [r['verdict'] for r in rows] == ['correct', 'malformed']

    def test_partial_submission_keeps_missing_assigned_question_missing_question_diagnostic(self, _case_partial_submission_keeps_missing_assigned_question):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_partial_submission_keeps_missing_assigned_question
        # Act
        rows, = run_case()
        # Assert
        assert rows[1]['malformed_kind'] == 'no_submission'

    @pytest.fixture
    def _case_unassigned_well_formed_id_is_not_alias_joined(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        path = _one_numeric_answers(tmp_path)

        def run_case(*, capture_error=False):
            row = score_submission('corebench', [{'task_id': 'corebench/capsule-1__hard__q99', 'answer': 0.9996}], answers=path)[0]
            return (row,)
        return run_case

    def test_unassigned_well_formed_id_is_not_alias_joined_malformed_verdict(self, _case_unassigned_well_formed_id_is_not_alias_joined):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_unassigned_well_formed_id_is_not_alias_joined
        # Act
        row, = run_case()
        # Assert
        assert row['verdict'] == 'malformed'

    def test_unassigned_well_formed_id_is_not_alias_joined_schema_diagnostic(self, _case_unassigned_well_formed_id_is_not_alias_joined):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_unassigned_well_formed_id_is_not_alias_joined
        # Act
        row, = run_case()
        # Assert
        assert row['malformed_kind'] == 'schema_invalid'

    @pytest.fixture
    def _case_explicit_public_assignment_selects_oracle_scope(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tids = ['corebench/capsule-1__hard__q0', 'corebench/capsule-1__hard__q1']
        path = _write_answers(tmp_path / 'answers.jsonl', [{'task_id': tid, 'answer': {'value': 5}} for tid in tids])

        def run_case(*, capture_error=False):
            rows = score_submission('corebench', [{'task_id': tids[0], 'answer': 5}], answers=path, expected_task_ids=[tids[0]], expected_answer_types={tids[0]: 'number'})
            return (rows,)
        return run_case

    def test_explicit_public_assignment_selects_oracle_scope(self, _case_explicit_public_assignment_selects_oracle_scope):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_explicit_public_assignment_selects_oracle_scope
        # Act
        rows, = run_case()
        # Assert
        assert len(rows) == 1 and rows[0]['verdict'] == 'correct'

    @pytest.fixture
    def _case_public_answer_type_not_derived_from_reference(self, tmp_path):
        """Shared isolated scenario; outcome assertions stay in test bodies."""
        tid = 'corebench/capsule-1__hard__q0'
        path = _one_numeric_answers(tmp_path)

        def run_case(*, capture_error=False):
            row = score_submission('corebench', [{'task_id': tid, 'answer': '0.9996'}], answers=path, expected_answer_types={tid: 'number'})[0]
            return (row,)
        return run_case

    def test_public_answer_type_not_derived_from_reference_malformed_verdict(self, _case_public_answer_type_not_derived_from_reference):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_public_answer_type_not_derived_from_reference
        # Act
        row, = run_case()
        # Assert
        assert row['verdict'] == 'malformed'

    def test_public_answer_type_not_derived_from_reference_schema_diagnostic(self, _case_public_answer_type_not_derived_from_reference):
        # Arrange: each scenario uses function-scoped inputs.
        run_case = _case_public_answer_type_not_derived_from_reference
        # Act
        row, = run_case()
        # Assert
        assert row['malformed_kind'] == 'schema_invalid'


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
