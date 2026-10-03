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

    def test_index_rejects_invalid_nonblank_json(self, tmp_path):
        # Arrange
        idx = tmp_path / 'index.jsonl'
        idx.write_text('\n{not json\n' + json.dumps({'task_ids': ['corebench/a__hard__q0']}) + '\n')
        # Act
        context = pytest.raises(_validate.PublicTaskContractError)
        # Assert
        with context:
            _validate.expected_task_ids_from_for_solver(tmp_path)

    def test_index_invalid_json_reports_assignment_error_kind(self, tmp_path):
        # Arrange
        from types import SimpleNamespace
        idx = tmp_path / 'index.jsonl'
        idx.write_text('\n{not json\n' + json.dumps({'task_ids': ['corebench/a__hard__q0']}) + '\n')
        # Act
        caught = None
        try:
            _validate.expected_task_ids_from_for_solver(tmp_path)
        except _validate.PublicTaskContractError as error:
            caught = SimpleNamespace(value=error)
        # Assert
        assert caught.value.errors[0]['kind'] == 'invalid_assignment'

    def test_index_invalid_json_preserves_original_line_number(self, tmp_path):
        # Arrange
        from types import SimpleNamespace
        idx = tmp_path / 'index.jsonl'
        idx.write_text('\n{not json\n' + json.dumps({'task_ids': ['corebench/a__hard__q0']}) + '\n')
        # Act
        caught = None
        try:
            _validate.expected_task_ids_from_for_solver(tmp_path)
        except _validate.PublicTaskContractError as error:
            caught = SimpleNamespace(value=error)
        # Assert
        assert caught.value.errors[0]['path'] == '$assignment[2]'


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
    def test_duplicate_cannot_hide_missing_peer_rejects_submission(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_A, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert result['ok'] is False

    def test_duplicate_cannot_hide_missing_peer_reports_duplicate_and_missing_ids(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_A, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert {e['kind'] for e in result['errors']} == {'duplicate_task_id', 'missing_task_id'}

    def test_duplicate_cannot_hide_missing_peer_names_missing_assigned_peer(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_A, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert next((e for e in result['errors'] if e['kind'] == 'missing_task_id'))['task_id'] == _NEW_B

    def test_unknown_cannot_replace_assigned_peer_rejects_submission(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_C, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert result['ok'] is False

    def test_unknown_cannot_replace_assigned_peer_reports_unknown_and_missing_ids(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_C, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert {e['kind'] for e in result['errors']} == {'unknown_task_id', 'missing_task_id'}

    def test_reordered_distinct_tasks_are_valid(self):
        # Arrange
        submission = [{'task_id': _NEW_B, 'answer': 0}, {'task_id': _NEW_A, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B])
        # Assert
        assert result == {'ok': True, 'errors': []}

    def test_legacy_id_never_aliases_new_question_id_rejects_submission(self):
        # Arrange
        submission = _valid_corebench_sub()
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A])
        # Assert
        assert result['ok'] is False

    def test_legacy_id_never_aliases_new_question_id_reports_exact_membership_errors(self):
        # Arrange
        submission = _valid_corebench_sub()
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A])
        # Assert
        assert {e['kind'] for e in result['errors']} == {'unknown_task_id', 'missing_task_id'}

    def test_duplicate_assignment_is_not_silently_deduplicated(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_A])
        # Assert
        assert any((e['kind'] == 'duplicate_assignment' for e in result['errors']))

    def test_empty_assignment_is_checked_accepts_empty_submission(self):
        # Arrange
        submission = []
        # Act
        actual = validate_submission('corebench', submission, expected_task_ids=[])
        # Assert
        assert actual == {'ok': True, 'errors': []}

    def test_empty_assignment_is_checked_rejects_unassigned_submission(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[])
        # Assert
        assert result['ok'] is False

    def test_empty_assignment_is_checked_reports_unassigned_task(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[])
        # Assert
        assert any((e['kind'] == 'unknown_task_id' for e in result['errors']))

    def test_host_partial_mode_leaves_missing_status_to_scorer(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A, _NEW_B], require_complete=False)
        # Assert
        assert result == {'ok': True, 'errors': []}

    def test_host_partial_mode_still_checks_membership_and_duplicates(self):
        # Arrange
        submission = [{'task_id': _NEW_C, 'answer': 0}, {'task_id': _NEW_C, 'answer': 1}]
        # Act
        result = validate_submission('corebench', submission, expected_task_ids=[_NEW_A], require_complete=False)
        # Assert
        assert {e['kind'] for e in result['errors']} == {'unknown_task_id', 'duplicate_task_id'}

    @pytest.mark.parametrize('task_id', ['corebench/capsule-1__question_a', 'corebench/__question_' + 'a' * 64, 'corebench/cap__question_' + 'g' * 64])
    def test_new_question_id_requires_native_and_full_hex_digest(self, task_id):
        # Arrange
        submission = [{'task_id': task_id, 'answer': 0}]
        # Act
        result = validate_submission('corebench', submission)
        # Assert
        assert any((e['kind'] == 'bad_task_id' for e in result['errors']))


class TestStrictPublicAnswerTypes:
    @pytest.mark.parametrize('answer', [0, -2, 1.5, '0.94', False, [0, None, 'x'], {'value': [0, True], 'metadata': None}])
    def test_undeclared_json_payloads_and_extra_fields_are_unchanged_accepts_json_answer(self, answer):
        # Arrange
        import copy
        sub = [{'task_id': _NEW_A, 'answer': answer, 'extra': {'keep': True}}]
        before = copy.deepcopy(sub)
        # Act
        result = validate_submission('corebench', sub)
        # Assert
        assert result['ok'] is True

    @pytest.mark.parametrize('answer', [0, -2, 1.5, '0.94', False, [0, None, 'x'], {'value': [0, True], 'metadata': None}])
    def test_undeclared_json_payloads_and_extra_fields_are_unchanged_warns_on_extra_field(self, answer):
        # Arrange
        import copy
        sub = [{'task_id': _NEW_A, 'answer': answer, 'extra': {'keep': True}}]
        before = copy.deepcopy(sub)
        # Act
        result = validate_submission('corebench', sub)
        # Assert
        assert [e['kind'] for e in result['errors']] == ['unknown_field']

    @pytest.mark.parametrize('answer', [0, -2, 1.5, '0.94', False, [0, None, 'x'], {'value': [0, True], 'metadata': None}])
    def test_undeclared_json_payloads_and_extra_fields_are_unchanged_preserves_entire_submission(self, answer):
        # Arrange
        import copy
        sub = [{'task_id': _NEW_A, 'answer': answer, 'extra': {'keep': True}}]
        before = copy.deepcopy(sub)
        # Act
        result = validate_submission('corebench', sub)
        # Assert
        assert sub == before

    @pytest.mark.parametrize('answer', [0, -2, 1.5, '0.94', False, [0, None, 'x'], {'value': [0, True], 'metadata': None}])
    def test_undeclared_json_payloads_and_extra_fields_are_unchanged_preserves_answer_python_type(self, answer):
        # Arrange
        import copy
        sub = [{'task_id': _NEW_A, 'answer': answer, 'extra': {'keep': True}}]
        before = copy.deepcopy(sub)
        # Act
        result = validate_submission('corebench', sub)
        # Assert
        assert type(sub[0]['answer']) is type(before[0]['answer'])

    @pytest.mark.parametrize('declared,answer', [('number', 0), ('number', -1.25), ('integer', 0), ('string', '0.94'), ('boolean', False), ('list', [0, {'x': True}]), ('object', {'x': [0, None]}), ('json', {'x': False})])
    def test_explicit_public_type_accepts_its_supported_value(self, declared, answer):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: declared})
        # Assert
        assert result == {'ok': True, 'errors': []}

    @pytest.mark.parametrize('declared,answer', [('number', '0.94'), ('number', True), ('integer', 1.0), ('integer', False), ('string', 1), ('boolean', 1), ('list', {'x': 0}), ('object', [0])])
    def test_declared_type_never_coerces_scientific_answer_rejects_submission(self, declared, answer):
        # Arrange
        sub = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', sub, expected_answer_types={_NEW_A: declared})
        # Assert
        assert result['ok'] is False

    @pytest.mark.parametrize('declared,answer', [('number', '0.94'), ('number', True), ('integer', 1.0), ('integer', False), ('string', 1), ('boolean', 1), ('list', {'x': 0}), ('object', [0])])
    def test_declared_type_never_coerces_scientific_answer_names_answer_type_field_error(self, declared, answer):
        # Arrange
        sub = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', sub, expected_answer_types={_NEW_A: declared})
        # Assert
        assert any((e['kind'] == 'wrong_answer_type' and e['field'] == 'answer' and (e['task_id'] == _NEW_A) for e in result['errors']))

    @pytest.mark.parametrize('declared,answer', [('number', '0.94'), ('number', True), ('integer', 1.0), ('integer', False), ('string', 1), ('boolean', 1), ('list', {'x': 0}), ('object', [0])])
    def test_declared_type_never_coerces_scientific_answer_preserves_answer_python_type(self, declared, answer):
        # Arrange
        sub = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', sub, expected_answer_types={_NEW_A: declared})
        # Assert
        assert type(sub[0]['answer']) is type(answer)

    @pytest.mark.parametrize('answer', [(1, 2), {1: 'x'}, {'nested': float('nan')}, float('inf')])
    def test_non_json_or_nonfinite_payloads_are_format_errors_rejects_submission(self, answer):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', submission)
        # Assert
        assert result['ok'] is False

    @pytest.mark.parametrize('answer', [(1, 2), {1: 'x'}, {'nested': float('nan')}, float('inf')])
    def test_non_json_or_nonfinite_payloads_are_format_errors_names_answer_format_path(self, answer):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': answer}]
        # Act
        result = validate_submission('corebench', submission)
        # Assert
        assert any((e['kind'] == 'wrong_type' and e['path'] == '$[0].answer' for e in result['errors']))

    def test_typed_null_preserves_original_reason_protocol(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': None, 'reason': 'agent abstained: unavailable input'}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'number'})
        # Assert
        assert result == {'ok': True, 'errors': []}

    def test_answered_reason_keeps_legacy_optional_json_behavior(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': 0, 'reason': {'note': 'kept'}}]
        # Act
        result = validate_submission('corebench', submission)
        # Assert
        assert result == {'ok': True, 'errors': []}

    def test_invalid_public_type_is_assignment_error_even_for_null_rejects_submission(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': None, 'reason': 'agent abstained: unavailable'}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'guess-from-reference'})
        # Assert
        assert result['ok'] is False

    def test_invalid_public_type_is_assignment_error_even_for_null_reports_assignment_error(self):
        # Arrange
        submission = [{'task_id': _NEW_A, 'answer': None, 'reason': 'agent abstained: unavailable'}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'guess-from-reference'})
        # Assert
        assert any((e['kind'] == 'invalid_assignment' for e in result['errors']))

    def test_feedback_never_echoes_submitted_answer_excludes_answer_payload(self):
        # Arrange
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        submission = [{'task_id': _NEW_A, 'answer': marker}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'number'})
        feedback = json.dumps(result)
        # Assert
        assert marker not in feedback

    def test_feedback_never_echoes_submitted_answer_names_declared_public_type(self):
        # Arrange
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        submission = [{'task_id': _NEW_A, 'answer': marker}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'number'})
        feedback = json.dumps(result)
        # Assert
        assert 'number' in feedback

    def test_feedback_never_echoes_submitted_answer_excludes_pydantic_input_field(self):
        # Arrange
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        submission = [{'task_id': _NEW_A, 'answer': marker}]
        # Act
        result = validate_submission('corebench', submission, expected_answer_types={_NEW_A: 'number'})
        # Assert
        assert 'input' not in result['errors'][0]


class TestPublicSelectedTaskContract:
    def test_selected_task_file_takes_precedence_over_catalog_accepts_assignment(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        (tmp_path / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_B]}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert contract['ok'] is True

    def test_selected_task_file_takes_precedence_over_catalog_labels_selected_scope(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        (tmp_path / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_B]}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert contract['scope'] == 'selected'

    def test_selected_task_file_takes_precedence_over_catalog_keeps_selected_task_ids(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        (tmp_path / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_B]}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert contract['task_ids'] == [_NEW_A]

    def test_selected_task_file_takes_precedence_over_catalog_keeps_public_answer_type(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        (tmp_path / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_B]}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert contract['answer_types'] == {_NEW_A: 'number'}

    def test_selected_task_file_takes_precedence_over_catalog_matches_public_id_helper(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        (tmp_path / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_B]}) + '\n')
        # Act
        actual = _validate.expected_task_ids_from_for_solver(tmp_path)
        # Assert
        assert actual == [_NEW_A]

    def test_absent_type_truthfully_defaults_to_json_accepts_assignment(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text(json.dumps({'task_id': _NEW_A, 'question': 'synthetic public question', 'input': {}, 'output': {}}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['ok'] is True

    def test_absent_type_truthfully_defaults_to_json_declares_json_default(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text(json.dumps({'task_id': _NEW_A, 'question': 'synthetic public question', 'input': {}, 'output': {}}) + '\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['answer_types'] == {_NEW_A: 'json'}

    @pytest.mark.parametrize('row', [[], {'task_id': 1}, {'task_id': _NEW_A, 'answer_type': 'numeric-from-oracle'}, {'wrong': 'field'}])
    def test_malformed_public_task_rows_are_attributed_rejects_assignment(self, tmp_path, row):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text(json.dumps(row) + '\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['ok'] is False

    @pytest.mark.parametrize('row', [[], {'task_id': 1}, {'task_id': _NEW_A, 'answer_type': 'numeric-from-oracle'}, {'wrong': 'field'}])
    def test_malformed_public_task_rows_are_attributed_reports_assignment_error_kind(self, tmp_path, row):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text(json.dumps(row) + '\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['errors'][0]['kind'] == 'invalid_assignment'

    @pytest.mark.parametrize('row', [[], {'task_id': 1}, {'task_id': _NEW_A, 'answer_type': 'numeric-from-oracle'}, {'wrong': 'field'}])
    def test_malformed_public_task_rows_are_attributed_names_original_assignment_row(self, tmp_path, row):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text(json.dumps(row) + '\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['errors'][0]['path'].startswith('$assignment[1]')

    @pytest.mark.parametrize('row', [[], {'task_ids': 'not-a-list'}, {'task_ids': [1]}, {}])
    def test_malformed_public_index_rows_are_attributed_rejects_assignment(self, tmp_path, row):
        # Arrange
        (tmp_path / 'index.jsonl').write_text(json.dumps(row) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert contract['ok'] is False

    @pytest.mark.parametrize('row', [[], {'task_ids': 'not-a-list'}, {'task_ids': [1]}, {}])
    def test_malformed_public_index_rows_are_attributed_reports_assignment_error(self, tmp_path, row):
        # Arrange
        (tmp_path / 'index.jsonl').write_text(json.dumps(row) + '\n')
        # Act
        contract = _validate.read_public_task_contract(tmp_path)
        # Assert
        assert any((e['kind'] == 'invalid_assignment' for e in contract['errors']))

    def test_duplicate_assignment_rows_are_not_lost_rejects_assignment(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text((json.dumps({'task_id': _NEW_A}) + '\n') * 2)
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['ok'] is False

    def test_duplicate_assignment_rows_are_not_lost_retains_duplicate_task_ids(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text((json.dumps({'task_id': _NEW_A}) + '\n') * 2)
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['task_ids'] == [_NEW_A, _NEW_A]

    def test_duplicate_assignment_rows_are_not_lost_reports_duplicate_assignment(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text((json.dumps({'task_id': _NEW_A}) + '\n') * 2)
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert any((e['kind'] == 'duplicate_assignment' for e in contract['errors']))

    def test_blank_lines_are_not_missing_or_bad_records_accepts_assignment(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text('\n\n' + json.dumps({'task_id': _NEW_A}) + '\n\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['ok'] is True

    def test_blank_lines_are_not_missing_or_bad_records_keeps_nonblank_task_id(self, tmp_path):
        # Arrange
        p = tmp_path / 'task.jsonl'
        p.write_text('\n\n' + json.dumps({'task_id': _NEW_A}) + '\n\n')
        # Act
        contract = _validate.read_public_task_contract(p)
        # Assert
        assert contract['task_ids'] == [_NEW_A]

    def test_explicit_empty_assignment_stays_empty(self, tmp_path):
        # Arrange
        (tmp_path / 'task.jsonl').write_text('\n')
        # Act
        actual = _validate.expected_task_ids_from_for_solver(tmp_path)
        # Assert
        assert actual == []

    def test_evaluator_filename_is_refused_without_reading_values_rejects_private_filename(self, tmp_path):
        # Arrange
        p = tmp_path / 'answers.jsonl'
        p.write_text('PRIVATE_ORACLE_SENTINEL_NOT_A_HINT')
        # Act
        result = _validate.read_public_task_contract(p)
        # Assert
        assert result['ok'] is False

    def test_evaluator_filename_is_refused_without_reading_values_excludes_private_file_contents(self, tmp_path):
        # Arrange
        p = tmp_path / 'answers.jsonl'
        p.write_text('PRIVATE_ORACLE_SENTINEL_NOT_A_HINT')
        # Act
        result = _validate.read_public_task_contract(p)
        # Assert
        assert 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT' not in json.dumps(result)


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

    def test_selected_scope_checks_only_selected_ids_returns_success_exit(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tasks)
        # Assert
        assert result.exit_code == 0, result.output

    def test_selected_scope_checks_only_selected_ids_reports_valid_submission(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tasks)
        parsed = json.loads(result.output)
        # Assert
        assert parsed['ok'] is True

    def test_selected_scope_checks_only_selected_ids_labels_selected_scope(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tasks)
        parsed = json.loads(result.output)
        # Assert
        assert parsed['validation_scope'] == 'selected'

    def test_selected_scope_checks_only_selected_ids_names_validation_contract(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tasks)
        parsed = json.loads(result.output)
        # Assert
        assert parsed['validation_contract'] == _validate.VALIDATION_CONTRACT

    def test_missing_default_metadata_explicitly_reports_shape_only_returns_success_exit(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        # Assert
        assert result.exit_code == 0, result.output

    def test_missing_default_metadata_explicitly_reports_shape_only_labels_shape_only_scope(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        parsed = json.loads(result.output)
        # Assert
        assert parsed['validation_scope'] == 'shape-only'

    def test_missing_default_metadata_explicitly_reports_shape_only_reports_shape_only_warning(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        parsed = json.loads(result.output)
        # Assert
        assert any((e['kind'] == 'shape_only' for e in parsed['errors']))

    def test_explicit_missing_assignment_is_hard_error_returns_failure_exit(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tmp_path / 'missing' / 'task.jsonl')
        # Assert
        assert result.exit_code == 1

    def test_explicit_missing_assignment_is_hard_error_reports_invalid_assignment(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tmp_path / 'missing' / 'task.jsonl')
        parsed = json.loads(result.output)
        # Assert
        assert parsed['ok'] is False

    def test_explicit_missing_assignment_is_hard_error_labels_invalid_scope(self, tmp_path):
        # Arrange
        # Inputs are supplied by this case and its function-scoped fixtures.
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}], tasks=tmp_path / 'missing' / 'task.jsonl')
        parsed = json.loads(result.output)
        # Assert
        assert parsed['validation_scope'] == 'invalid-assignment'

    def test_malformed_default_index_never_becomes_shape_only_returns_failure_exit(self, tmp_path):
        # Arrange
        directory = tmp_path / 'ai-for-science' / 'corebench' / 'for_solver'
        directory.mkdir(parents=True)
        (directory / 'index.jsonl').write_text('{not json\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        # Assert
        assert result.exit_code == 1

    def test_malformed_default_index_never_becomes_shape_only_labels_invalid_scope(self, tmp_path):
        # Arrange
        directory = tmp_path / 'ai-for-science' / 'corebench' / 'for_solver'
        directory.mkdir(parents=True)
        (directory / 'index.jsonl').write_text('{not json\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        parsed = json.loads(result.output)
        # Assert
        assert parsed['validation_scope'] == 'invalid-assignment'

    def test_malformed_default_index_never_becomes_shape_only_omits_shape_only_warning(self, tmp_path):
        # Arrange
        directory = tmp_path / 'ai-for-science' / 'corebench' / 'for_solver'
        directory.mkdir(parents=True)
        (directory / 'index.jsonl').write_text('{not json\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}])
        parsed = json.loads(result.output)
        # Assert
        assert not any((e['kind'] == 'shape_only' for e in parsed['errors']))

    def test_public_numeric_type_repair_feedback_is_oracle_free_returns_failure_exit(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': marker}], tasks=tasks)
        # Assert
        assert result.exit_code == 1

    def test_public_numeric_type_repair_feedback_is_oracle_free_reports_wrong_answer_type(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': marker}], tasks=tasks)
        parsed = json.loads(result.output)
        # Assert
        assert any((e['kind'] == 'wrong_answer_type' for e in parsed['errors']))

    def test_public_numeric_type_repair_feedback_is_oracle_free_excludes_answer_payload(self, tmp_path):
        # Arrange
        tasks = tmp_path / 'task.jsonl'
        tasks.write_text(json.dumps({'task_id': _NEW_A, 'answer_type': 'number'}) + '\n')
        marker = 'PRIVATE_ORACLE_SENTINEL_NOT_A_HINT'
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': marker}], tasks=tasks)
        # Assert
        assert marker not in result.output

    def test_default_catalog_scope_is_truthfully_labeled_returns_success_exit(self, tmp_path):
        # Arrange
        directory = tmp_path / 'ai-for-science' / 'corebench' / 'for_solver'
        directory.mkdir(parents=True)
        (directory / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_A, _NEW_B]}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_B, 'answer': 1}])
        # Assert
        assert result.exit_code == 0, result.output

    def test_default_catalog_scope_is_truthfully_labeled_labels_catalog_scope(self, tmp_path):
        # Arrange
        directory = tmp_path / 'ai-for-science' / 'corebench' / 'for_solver'
        directory.mkdir(parents=True)
        (directory / 'index.jsonl').write_text(json.dumps({'task_ids': [_NEW_A, _NEW_B]}) + '\n')
        # Act
        result = self.invoke(tmp_path, [{'task_id': _NEW_A, 'answer': 0}, {'task_id': _NEW_B, 'answer': 1}])
        # Assert
        assert json.loads(result.output)['validation_scope'] == 'catalog'


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
