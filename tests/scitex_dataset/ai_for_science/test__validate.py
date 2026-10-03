#!/usr/bin/env python3
"""Tests for ai_for_science._validate (structural, ORACLE-FREE validator).

Checks use real submission objects and synthetic public assignment files.
CLI controls use real Click dispatch without an oracle, download or scorer.
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

    def test_index_reports_bad_lines_and_ignores_blank_separators(self, tmp_path):
        # Malformed present metadata must not silently disable membership.
        idx = tmp_path / "index.jsonl"
        idx.write_text(
            "\n{not json\n" + json.dumps({"task_ids": ["corebench/a__hard__q0"]}) + "\n"
        )
        with pytest.raises(_validate.PublicTaskContractError) as caught:
            _validate.expected_task_ids_from_for_solver(tmp_path)
        assert caught.value.errors[0]["kind"] == "invalid_assignment"
        assert caught.value.errors[0]["path"] == "$assignment[2]"


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


# Additive operator320/324 controls: public identity/type feedback only.
_NEW_A = "corebench/capsule-1__question_" + "a" * 64
_NEW_B = "corebench/capsule-1__question_" + "b" * 64
_NEW_C = "corebench/capsule-1__question_" + "c" * 64


class TestExactSelectedMembership:
    def test_duplicate_cannot_hide_missing_peer(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 0}, {"task_id": _NEW_A, "answer": 1}], expected_task_ids=[_NEW_A, _NEW_B])
        assert result["ok"] is False
        assert {e["kind"] for e in result["errors"]} == {"duplicate_task_id", "missing_task_id"}
        assert next(e for e in result["errors"] if e["kind"] == "missing_task_id")["task_id"] == _NEW_B

    def test_unknown_cannot_replace_assigned_peer(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 0}, {"task_id": _NEW_C, "answer": 1}], expected_task_ids=[_NEW_A, _NEW_B])
        assert result["ok"] is False
        assert {e["kind"] for e in result["errors"]} == {"unknown_task_id", "missing_task_id"}

    def test_reordered_distinct_tasks_are_valid(self):
        result = validate_submission("corebench", [{"task_id": _NEW_B, "answer": 0}, {"task_id": _NEW_A, "answer": 1}], expected_task_ids=[_NEW_A, _NEW_B])
        assert result == {"ok": True, "errors": []}

    def test_legacy_id_never_aliases_new_question_id(self):
        result = validate_submission("corebench", _valid_corebench_sub(), expected_task_ids=[_NEW_A])
        assert result["ok"] is False
        assert {e["kind"] for e in result["errors"]} == {"unknown_task_id", "missing_task_id"}

    def test_duplicate_assignment_is_not_silently_deduplicated(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 1}], expected_task_ids=[_NEW_A, _NEW_A])
        assert any(e["kind"] == "duplicate_assignment" for e in result["errors"])

    def test_empty_assignment_is_checked(self):
        assert validate_submission("corebench", [], expected_task_ids=[]) == {"ok": True, "errors": []}
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 0}], expected_task_ids=[])
        assert result["ok"] is False
        assert any(e["kind"] == "unknown_task_id" for e in result["errors"])

    def test_host_partial_mode_leaves_missing_status_to_scorer(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 0}], expected_task_ids=[_NEW_A, _NEW_B], require_complete=False)
        assert result == {"ok": True, "errors": []}

    def test_host_partial_mode_still_checks_membership_and_duplicates(self):
        result = validate_submission("corebench", [{"task_id": _NEW_C, "answer": 0}, {"task_id": _NEW_C, "answer": 1}], expected_task_ids=[_NEW_A], require_complete=False)
        assert {e["kind"] for e in result["errors"]} == {"unknown_task_id", "duplicate_task_id"}

    @pytest.mark.parametrize("task_id", ["corebench/capsule-1__question_a", "corebench/__question_" + "a" * 64, "corebench/cap__question_" + "g" * 64])
    def test_new_question_id_requires_native_and_full_hex_digest(self, task_id):
        result = validate_submission("corebench", [{"task_id": task_id, "answer": 0}])
        assert any(e["kind"] == "bad_task_id" for e in result["errors"])


class TestStrictPublicAnswerTypes:
    @pytest.mark.parametrize("answer", [0, -2, 1.5, "0.94", False, [0, None, "x"], {"value": [0, True], "metadata": None}])
    def test_undeclared_json_payloads_and_extra_fields_are_unchanged(self, answer):
        import copy
        sub = [{"task_id": _NEW_A, "answer": answer, "extra": {"keep": True}}]
        before = copy.deepcopy(sub)
        result = validate_submission("corebench", sub)
        assert result["ok"] is True
        assert [e["kind"] for e in result["errors"]] == ["unknown_field"]
        assert sub == before
        assert type(sub[0]["answer"]) is type(before[0]["answer"])

    @pytest.mark.parametrize("declared,answer", [("number", 0), ("number", -1.25), ("integer", 0), ("string", "0.94"), ("boolean", False), ("list", [0, {"x": True}]), ("object", {"x": [0, None]}), ("json", {"x": False})])
    def test_explicit_public_type_accepts_its_supported_value(self, declared, answer):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": answer}], expected_answer_types={_NEW_A: declared})
        assert result == {"ok": True, "errors": []}

    @pytest.mark.parametrize("declared,answer", [("number", "0.94"), ("number", True), ("integer", 1.0), ("integer", False), ("string", 1), ("boolean", 1), ("list", {"x": 0}), ("object", [0])])
    def test_declared_type_never_coerces_scientific_answer(self, declared, answer):
        sub = [{"task_id": _NEW_A, "answer": answer}]
        result = validate_submission("corebench", sub, expected_answer_types={_NEW_A: declared})
        assert result["ok"] is False
        assert any(e["kind"] == "wrong_answer_type" and e["field"] == "answer" and e["task_id"] == _NEW_A for e in result["errors"])
        assert type(sub[0]["answer"]) is type(answer)

    @pytest.mark.parametrize("answer", [(1, 2), {1: "x"}, {"nested": float("nan")}, float("inf")])
    def test_non_json_or_nonfinite_payloads_are_format_errors(self, answer):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": answer}])
        assert result["ok"] is False
        assert any(e["kind"] == "wrong_type" and e["path"] == "$[0].answer" for e in result["errors"])

    def test_typed_null_preserves_original_reason_protocol(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": None, "reason": "agent abstained: unavailable input"}], expected_answer_types={_NEW_A: "number"})
        assert result == {"ok": True, "errors": []}

    def test_answered_reason_keeps_legacy_optional_json_behavior(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": 0, "reason": {"note": "kept"}}])
        assert result == {"ok": True, "errors": []}

    def test_invalid_public_type_is_assignment_error_even_for_null(self):
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": None, "reason": "agent abstained: unavailable"}], expected_answer_types={_NEW_A: "guess-from-reference"})
        assert result["ok"] is False
        assert any(e["kind"] == "invalid_assignment" for e in result["errors"])

    def test_feedback_never_echoes_submitted_answer(self):
        marker = "PRIVATE_ORACLE_SENTINEL_NOT_A_HINT"
        result = validate_submission("corebench", [{"task_id": _NEW_A, "answer": marker}], expected_answer_types={_NEW_A: "number"})
        feedback = json.dumps(result)
        assert marker not in feedback
        assert "number" in feedback
        assert "input" not in result["errors"][0]


class TestPublicSelectedTaskContract:
    def test_selected_task_file_takes_precedence_over_catalog(self, tmp_path):
        (tmp_path / "task.jsonl").write_text(json.dumps({"task_id": _NEW_A, "answer_type": "number"}) + "\n")
        (tmp_path / "index.jsonl").write_text(json.dumps({"task_ids": [_NEW_B]}) + "\n")
        contract = _validate.read_public_task_contract(tmp_path)
        assert contract["ok"] is True
        assert contract["scope"] == "selected"
        assert contract["task_ids"] == [_NEW_A]
        assert contract["answer_types"] == {_NEW_A: "number"}
        assert _validate.expected_task_ids_from_for_solver(tmp_path) == [_NEW_A]

    def test_absent_type_truthfully_defaults_to_json(self, tmp_path):
        p = tmp_path / "task.jsonl"
        p.write_text(json.dumps({"task_id": _NEW_A, "question": "synthetic public question", "input": {}, "output": {}}) + "\n")
        contract = _validate.read_public_task_contract(p)
        assert contract["ok"] is True
        assert contract["answer_types"] == {_NEW_A: "json"}

    @pytest.mark.parametrize("row", [[], {"task_id": 1}, {"task_id": _NEW_A, "answer_type": "numeric-from-oracle"}, {"wrong": "field"}])
    def test_malformed_public_task_rows_are_attributed(self, tmp_path, row):
        p = tmp_path / "task.jsonl"
        p.write_text(json.dumps(row) + "\n")
        contract = _validate.read_public_task_contract(p)
        assert contract["ok"] is False
        assert contract["errors"][0]["kind"] == "invalid_assignment"
        assert contract["errors"][0]["path"].startswith("$assignment[1]")

    @pytest.mark.parametrize("row", [[], {"task_ids": "not-a-list"}, {"task_ids": [1]}, {}])
    def test_malformed_public_index_rows_are_attributed(self, tmp_path, row):
        (tmp_path / "index.jsonl").write_text(json.dumps(row) + "\n")
        contract = _validate.read_public_task_contract(tmp_path)
        assert contract["ok"] is False
        assert any(e["kind"] == "invalid_assignment" for e in contract["errors"])

    def test_duplicate_assignment_rows_are_not_lost(self, tmp_path):
        p = tmp_path / "task.jsonl"
        p.write_text((json.dumps({"task_id": _NEW_A}) + "\n") * 2)
        contract = _validate.read_public_task_contract(p)
        assert contract["ok"] is False
        assert contract["task_ids"] == [_NEW_A, _NEW_A]
        assert any(e["kind"] == "duplicate_assignment" for e in contract["errors"])

    def test_blank_lines_are_not_missing_or_bad_records(self, tmp_path):
        p = tmp_path / "task.jsonl"
        p.write_text("\n\n" + json.dumps({"task_id": _NEW_A}) + "\n\n")
        contract = _validate.read_public_task_contract(p)
        assert contract["ok"] is True
        assert contract["task_ids"] == [_NEW_A]

    def test_explicit_empty_assignment_stays_empty(self, tmp_path):
        (tmp_path / "task.jsonl").write_text("\n")
        assert _validate.expected_task_ids_from_for_solver(tmp_path) == []

    def test_evaluator_filename_is_refused_without_reading_values(self, tmp_path):
        p = tmp_path / "answers.jsonl"
        p.write_text("PRIVATE_ORACLE_SENTINEL_NOT_A_HINT")
        result = _validate.read_public_task_contract(p)
        assert result["ok"] is False
        assert "PRIVATE_ORACLE_SENTINEL_NOT_A_HINT" not in json.dumps(result)


class TestValidateCli:
    @staticmethod
    def invoke(tmp_path, sub, *, tasks=None):
        from types import SimpleNamespace
        from click.testing import CliRunner
        from scitex_dataset._cli._agentic import _make_validate_command
        p = tmp_path / "submission.json"
        p.write_text(json.dumps(sub))
        args = ["--dataset-root", str(tmp_path), "--submission", str(p), "--json"]
        if tasks is not None:
            args += ["--tasks", str(tasks)]
        return CliRunner().invoke(_make_validate_command("corebench", SimpleNamespace(BENCHMARK="corebench")), args)

    def test_selected_scope_checks_only_selected_ids(self, tmp_path):
        tasks = tmp_path / "task.jsonl"
        tasks.write_text(json.dumps({"task_id": _NEW_A}) + "\n")
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": 0}], tasks=tasks)
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["ok"] is True
        assert parsed["validation_scope"] == "selected"
        assert parsed["validation_contract"] == _validate.VALIDATION_CONTRACT

    def test_missing_default_metadata_explicitly_reports_shape_only(self, tmp_path):
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": 0}])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["validation_scope"] == "shape-only"
        assert any(e["kind"] == "shape_only" for e in parsed["errors"])

    def test_explicit_missing_assignment_is_hard_error(self, tmp_path):
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": 0}], tasks=tmp_path / "missing" / "task.jsonl")
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["ok"] is False
        assert parsed["validation_scope"] == "invalid-assignment"

    def test_malformed_default_index_never_becomes_shape_only(self, tmp_path):
        directory = tmp_path / "ai-for-science" / "corebench" / "for_solver"
        directory.mkdir(parents=True)
        (directory / "index.jsonl").write_text("{not json\n")
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": 0}])
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["validation_scope"] == "invalid-assignment"
        assert not any(e["kind"] == "shape_only" for e in parsed["errors"])

    def test_public_numeric_type_repair_feedback_is_oracle_free(self, tmp_path):
        tasks = tmp_path / "task.jsonl"
        tasks.write_text(json.dumps({"task_id": _NEW_A, "answer_type": "number"}) + "\n")
        marker = "PRIVATE_ORACLE_SENTINEL_NOT_A_HINT"
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": marker}], tasks=tasks)
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert any(e["kind"] == "wrong_answer_type" for e in parsed["errors"])
        assert marker not in result.output

    def test_default_catalog_scope_is_truthfully_labeled(self, tmp_path):
        directory = tmp_path / "ai-for-science" / "corebench" / "for_solver"
        directory.mkdir(parents=True)
        (directory / "index.jsonl").write_text(json.dumps({"task_ids": [_NEW_A, _NEW_B]}) + "\n")
        result = self.invoke(tmp_path, [{"task_id": _NEW_A, "answer": 0}, {"task_id": _NEW_B, "answer": 1}])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["validation_scope"] == "catalog"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
