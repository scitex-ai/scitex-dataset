#!/usr/bin/env python3
"""Tests for ai_for_science._validate (structural, ORACLE-FREE validator).

No mocks / monkeypatch: every check uses real submission objects and,
for the oracle-free proof, a real tmp dir with NO answers.jsonl present.
"""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _validate
from scitex_dataset.ai_for_science._validate import validate_submission


def _valid_corebench_sub():
    return [{"task_id": "corebench/capsule-1111111__hard__q0", "answer": 0.81}]


# ---------------------------------------------------------------------------
# Oracle-free by construction
# ---------------------------------------------------------------------------


class TestOracleFree:
    def test_module_does_not_expose_scorer(self):
        # Arrange
        module = _validate
        # Act
        exposes_scorer = hasattr(module, "score_submission")
        # Assert
        assert exposes_scorer is False

    def test_source_never_imports_score(self):
        # Arrange
        src = Path(_validate.__file__).read_text()
        # Act
        imports_score = "import _score" in src or "from ._score" in src
        # Assert
        assert imports_score is False

    def test_validates_in_dir_with_no_answers_jsonl(self, tmp_path):
        # Arrange — an empty tmp dir: no oracle answers.jsonl exists anywhere.
        oracle_absent = not (tmp_path / "answers.jsonl").exists()
        sub = _valid_corebench_sub()
        # Act
        result = validate_submission("corebench", sub) if oracle_absent else None
        # Assert
        assert result["ok"] is True

    def test_valid_submission_has_no_errors(self):
        # Arrange
        sub = _valid_corebench_sub()
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Error kinds
# ---------------------------------------------------------------------------


class TestErrorKinds:
    def test_non_array_top_level_is_wrong_type(self):
        # Arrange
        sub = {"task_id": "corebench/x__hard__q0", "answer": 1}
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "wrong_type" for e in result["errors"])

    def test_non_array_top_level_is_not_ok(self):
        # Arrange
        sub = {"task_id": "corebench/x__hard__q0", "answer": 1}
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is False

    def test_missing_task_id_is_missing_field(self):
        # Arrange
        sub = [{"answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "missing_field" for e in result["errors"])

    def test_missing_answer_is_missing_field(self):
        # Arrange
        sub = [{"task_id": "corebench/capsule-1__hard__q0"}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(
            e["kind"] == "missing_field" and "answer" in e["message"]
            for e in result["errors"]
        )

    def test_non_string_task_id_is_wrong_type(self):
        # Arrange
        sub = [{"task_id": 123, "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "wrong_type" for e in result["errors"])

    def test_bad_prefix_is_bad_task_id(self):
        # Arrange — wrong benchmark prefix.
        sub = [{"task_id": "bixbench/x", "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])

    def test_corebench_bad_shape_is_bad_task_id(self):
        # Arrange — right prefix, wrong __diff__qN shape.
        sub = [{"task_id": "corebench/capsule-1", "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])

    def test_bixbench_prefix_only_is_valid(self):
        # Arrange — non-corebench benchmarks only need the prefix.
        sub = [{"task_id": "bixbench/short-id-x", "answer": "a"}]
        # Act
        result = validate_submission("bixbench", sub)
        # Assert
        assert result["ok"] is True

    def test_wrong_count_flagged_against_expected(self):
        # Arrange — one item but two expected.
        sub = _valid_corebench_sub()
        expected = ["corebench/a__hard__q0", "corebench/b__hard__q0"]
        # Act
        result = validate_submission("corebench", sub, expected_task_ids=expected)
        # Assert
        assert any(e["kind"] == "wrong_count" for e in result["errors"])

    def test_unknown_field_is_reported(self):
        # Arrange — an extra key beyond task_id/answer/reason.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": 1, "confidence": 0.9}
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "unknown_field" for e in result["errors"])

    def test_unknown_field_does_not_flip_ok(self):
        # Arrange — a warn-only kind must not make the submission invalid.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": 1, "confidence": 0.9}
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is True

    def test_reason_field_is_not_unknown(self):
        # Arrange — ``reason`` is a known (abstention) key.
        sub = [
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": None,
                "reason": "agent abstained: unclear",
            }
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert not any(e["kind"] == "unknown_field" for e in result["errors"])


# ---------------------------------------------------------------------------
# Path inputs
# ---------------------------------------------------------------------------


class TestPathInputs:
    def test_missing_file_is_no_file(self, tmp_path):
        # Arrange
        missing = tmp_path / "nope.json"
        # Act
        result = validate_submission("corebench", missing)
        # Assert
        assert result["errors"][0]["kind"] == "no_file"

    def test_non_json_file_is_unparseable(self, tmp_path):
        # Arrange
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        # Act
        result = validate_submission("corebench", bad)
        # Assert
        assert result["errors"][0]["kind"] == "unparseable"

    def test_valid_file_path_ok(self, tmp_path):
        # Arrange
        good = tmp_path / "good.json"
        good.write_text(json.dumps(_valid_corebench_sub()))
        # Act
        result = validate_submission("corebench", good)
        # Assert
        assert result["ok"] is True


class TestExpectedTaskIdsFromForSolver:
    def test_reads_task_ids_from_index(self, tmp_path):
        # Arrange — a for_solver index.jsonl (agent-visible, not the oracle).
        idx = tmp_path / "index.jsonl"
        idx.write_text(
            json.dumps({"task_ids": ["corebench/a__hard__q0"]})
            + "\n"
            + json.dumps({"task_ids": ["corebench/b__hard__q0"]})
            + "\n"
        )
        # Act
        ids = _validate.expected_task_ids_from_for_solver(tmp_path)
        # Assert
        assert ids == ["corebench/a__hard__q0", "corebench/b__hard__q0"]

    def test_missing_index_returns_none(self, tmp_path):
        # Arrange
        empty = tmp_path
        # Act
        ids = _validate.expected_task_ids_from_for_solver(empty)
        # Assert
        assert ids is None

    def test_index_skips_blank_and_bad_lines(self, tmp_path):
        # Arrange — a blank line and a malformed JSON line are skipped.
        idx = tmp_path / "index.jsonl"
        idx.write_text(
            "\n{not json\n" + json.dumps({"task_ids": ["corebench/a__hard__q0"]}) + "\n"
        )
        # Act
        ids = _validate.expected_task_ids_from_for_solver(tmp_path)
        # Assert
        assert ids == ["corebench/a__hard__q0"]


class TestTaskIdEdgeShapes:
    def test_prefix_only_empty_rest_is_bad(self):
        # Arrange — the prefix is present but there is no id after it.
        sub = [{"task_id": "corebench/", "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])

    def test_empty_native_segment_is_bad(self):
        # Arrange — the <native> segment before __ is empty.
        sub = [{"task_id": "corebench/__hard__q0", "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])

    def test_non_qN_question_segment_is_bad(self):
        # Arrange — the trailing segment must be q<digits>.
        sub = [{"task_id": "corebench/cap__hard__x0", "answer": 1}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])

    def test_non_object_item_is_wrong_type(self):
        # Arrange — a bare scalar where an object is required.
        sub = [123]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "wrong_type" for e in result["errors"])


# ---------------------------------------------------------------------------
# Honest abstention — a null answer MUST carry a non-empty reason
# ---------------------------------------------------------------------------


class TestReasonOnNull:
    """The submission contract: ``answer: null`` requires a non-empty
    ``reason`` (honest abstention; silent no-answer is forbidden)."""

    def test_null_answer_with_reason_is_ok(self):
        # Arrange — (a) the honest-abstention case: null + actionable reason.
        sub = [
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": None,
                "reason": "agent abstained: OCR fallback failed on figure 3",
            }
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is True

    def test_null_answer_missing_reason_is_not_ok(self):
        # Arrange — (b) null answer with NO reason key at all.
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": None}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is False

    def test_null_answer_missing_reason_kind_is_missing_reason(self):
        # Arrange — (b) the finding kind names the reason-on-null rule.
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": None}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(e["kind"] == "missing_reason" for e in result["errors"])

    def test_null_answer_missing_reason_message_names_question(self):
        # Arrange — the per-entry finding names the offending task_id.
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": None}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert any(
            e["kind"] == "missing_reason"
            and "corebench/capsule-1__hard__q0" in e["message"]
            for e in result["errors"]
        )

    def test_null_answer_empty_reason_is_not_ok(self):
        # Arrange — (c) reason present but empty string.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": None, "reason": ""}
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is False

    def test_null_answer_whitespace_reason_is_not_ok(self):
        # Arrange — (c) reason is whitespace-only → empty after strip().
        sub = [
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": None,
                "reason": "   \t\n",
            }
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is False

    def test_null_answer_null_reason_is_not_ok(self):
        # Arrange — (c) reason present but null.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": None, "reason": None}
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is False

    def test_non_null_answer_without_reason_is_ok(self):
        # Arrange — (d) an answered claim never needs a reason.
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": "0.94"}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is True

    def test_non_null_answer_without_reason_has_no_missing_reason(self):
        # Arrange — (d) reason stays optional for answered claims.
        sub = [{"task_id": "corebench/capsule-1__hard__q0", "answer": "0.94"}]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert not any(e["kind"] == "missing_reason" for e in result["errors"])

    def test_mixed_answered_and_null_with_reason_is_ok(self):
        # Arrange — (e) a fully-valid mixed submission.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": "0.94"},
            {
                "task_id": "corebench/capsule-1__hard__q1",
                "answer": None,
                "reason": "agent abstained: source file not reproducible",
            },
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        assert result["ok"] is True

    def test_null_answer_missing_reason_flags_only_the_offender(self):
        # Arrange — (e-inverse) only the reasonless null is flagged, once.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": "0.94"},
            {"task_id": "corebench/capsule-1__hard__q1", "answer": None},
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        offenders = [e for e in result["errors"] if e["kind"] == "missing_reason"]
        assert len(offenders) == 1

    def test_null_answer_missing_reason_offender_names_that_entry(self):
        # Arrange — the single offender's message names the reasonless task_id.
        sub = [
            {"task_id": "corebench/capsule-1__hard__q0", "answer": "0.94"},
            {"task_id": "corebench/capsule-1__hard__q1", "answer": None},
        ]
        # Act
        result = validate_submission("corebench", sub)
        # Assert
        offenders = [e for e in result["errors"] if e["kind"] == "missing_reason"]
        assert "corebench/capsule-1__hard__q1" in offenders[0]["message"]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
