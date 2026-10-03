#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/ai_for_science/test__gate.py

"""Tests for the pure, scitex_dev-AGNOSTIC gate logic (``_gate``).

Every case builds a REAL temp capsule workdir (a ``task.jsonl`` and a
``submission.json``) and exercises :func:`build_gate_result` directly —
no ``scitex_dev`` needed, so these tests carry the coverage.
"""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _gate
from scitex_dataset.ai_for_science._gate import build_gate_result

_TASK_ID_A = "corebench/capsule-1111111__hard__q0"
_TASK_ID_B = "corebench/capsule-2222222__easy__q1"


def _write_capsule(workdir: Path, task_ids=(_TASK_ID_A, _TASK_ID_B)):
    """Write a real 2-row corebench ``task.jsonl`` into ``workdir``."""
    lines = [
        json.dumps({"benchmark": "corebench", "task_id": tid}) for tid in task_ids
    ]
    (workdir / "task.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_submission(workdir: Path, items):
    """Write items to the CANONICAL ``submission/submission.json`` path."""
    sub_dir = workdir / "submission"
    sub_dir.mkdir(exist_ok=True)
    (sub_dir / "submission.json").write_text(json.dumps(items), encoding="utf-8")


def _valid_items():
    return [
        {"task_id": _TASK_ID_A, "answer": 0.81},
        {"task_id": _TASK_ID_B, "answer": "cat"},
    ]


def test_valid_submission_passes(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    _write_submission(tmp_path, _valid_items())
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_valid_submission_has_no_error_findings(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    _write_submission(tmp_path, _valid_items())
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert [f for f in result["findings"] if f["severity"] == "error"] == []


def test_missing_submission_file_fails(tmp_path):
    # Arrange — capsule present but no submission.json written.
    _write_capsule(tmp_path)
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is False


def test_missing_submission_file_reports_no_file_kind(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert kinds == ["no_file"]


def test_missing_submission_file_has_non_empty_fix_hint(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    # Act
    finding = build_gate_result(tmp_path, {})["findings"][0]
    # Assert
    assert finding["fix_hint"] != ""


def test_unparseable_submission_reports_unparseable_kind(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "submission.json").write_text(
        "{not json", encoding="utf-8"
    )
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "unparseable" in kinds


def test_wrong_count_submission_reports_wrong_count_kind(tmp_path):
    # Arrange — one item for a two-task capsule.
    _write_capsule(tmp_path)
    _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 1}])
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "wrong_count" in kinds


def test_bad_task_id_submission_reports_bad_task_id_kind(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    _write_submission(
        tmp_path,
        [
            {"task_id": "corebench/not-a-real-shape", "answer": 1},
            {"task_id": _TASK_ID_B, "answer": 2},
        ],
    )
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "bad_task_id" in kinds


def test_bad_task_id_finding_has_error_severity(tmp_path):
    # Arrange
    _write_capsule(tmp_path)
    _write_submission(
        tmp_path,
        [
            {"task_id": "corebench/not-a-real-shape", "answer": 1},
            {"task_id": _TASK_ID_B, "answer": 2},
        ],
    )
    # Act
    findings = build_gate_result(tmp_path, {})["findings"]
    bad = next(f for f in findings if f["kind"] == "bad_task_id")
    # Assert
    assert bad["severity"] == "error"


def test_missing_field_submission_reports_missing_field_kind(tmp_path):
    # Arrange — second item lacks 'answer'.
    _write_capsule(tmp_path)
    _write_submission(
        tmp_path,
        [{"task_id": _TASK_ID_A, "answer": 1}, {"task_id": _TASK_ID_B}],
    )
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "missing_field" in kinds


def test_unknown_field_finding_has_warning_severity(tmp_path):
    # Arrange — an extra 'confidence' key is an unknown-field WARNING.
    _write_capsule(tmp_path)
    _write_submission(
        tmp_path,
        [
            {"task_id": _TASK_ID_A, "answer": 1, "confidence": 0.9},
            {"task_id": _TASK_ID_B, "answer": 2},
        ],
    )
    # Act
    findings = build_gate_result(tmp_path, {})["findings"]
    unknown = next(f for f in findings if f["kind"] == "unknown_field")
    # Assert
    assert unknown["severity"] == "warning"


def test_unknown_field_only_submission_still_passes(tmp_path):
    # Arrange — only a non-fatal unknown-field WARNING present.
    _write_capsule(tmp_path)
    _write_submission(
        tmp_path,
        [
            {"task_id": _TASK_ID_A, "answer": 1, "confidence": 0.9},
            {"task_id": _TASK_ID_B, "answer": 2},
        ],
    )
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_configurable_submission_filename_is_used(tmp_path):
    # Arrange — answers live in a custom filename via config.
    _write_capsule(tmp_path)
    (tmp_path / "answers.json").write_text(
        json.dumps(_valid_items()), encoding="utf-8"
    )
    # Act
    result = build_gate_result(tmp_path, {"submission_file": "answers.json"})
    # Assert
    assert result["passed"] is True


def test_root_submission_json_fallback_resolves(tmp_path):
    # Arrange — no submission/ subdir; tolerant root fallback file present.
    _write_capsule(tmp_path)
    (tmp_path / "submission.json").write_text(
        json.dumps(_valid_items()), encoding="utf-8"
    )
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_canonical_submission_subdir_is_default(tmp_path):
    # Arrange — answers only at the canonical submission/submission.json path.
    _write_capsule(tmp_path)
    _write_submission(tmp_path, _valid_items())
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_missing_submission_no_file_hint_names_canonical_path(tmp_path):
    # Arrange — nothing written; the no_file hint must name submission/.
    _write_capsule(tmp_path)
    # Act
    hint = build_gate_result(tmp_path, {})["findings"][0]["fix_hint"]
    # Assert
    assert "submission/submission.json" in hint


def test_capsule_subdir_is_discovered(tmp_path):
    # Arrange — task.jsonl + submission live under a capsule-NNN/ subdir.
    capsule = tmp_path / "capsule-0424"
    capsule.mkdir()
    _write_capsule(capsule)
    _write_submission(capsule, _valid_items())
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_no_capsule_no_benchmark_suppresses_bad_task_id(tmp_path):
    # Arrange — no task.jsonl anywhere, no benchmark in config.
    _write_submission(
        tmp_path, [{"task_id": "anything", "answer": 1}]
    )
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "bad_task_id" not in kinds


def test_no_capsule_no_benchmark_emits_benchmark_unknown_info(tmp_path):
    # Arrange
    _write_submission(tmp_path, [{"task_id": "anything", "answer": 1}])
    # Act
    findings = build_gate_result(tmp_path, {})["findings"]
    info = [f for f in findings if f["kind"] == "benchmark_unknown"]
    # Assert
    assert len(info) == 1


def test_no_capsule_valid_shape_submission_passes(tmp_path):
    # Arrange — structurally valid array, benchmark unknown.
    _write_submission(tmp_path, [{"task_id": "anything", "answer": 1}])
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is True


def test_no_capsule_structure_error_still_surfaces(tmp_path):
    # Arrange — a top-level object (not an array) is a hard structure error.
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "submission.json").write_text(
        json.dumps({"task_id": "x", "answer": 1}), encoding="utf-8"
    )
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert "wrong_type" in kinds


def test_fail_closed_on_corrupt_task_jsonl(tmp_path):
    # Arrange — a task.jsonl with a non-JSON line makes row parsing raise.
    (tmp_path / "task.jsonl").write_text("{not json at all\n", encoding="utf-8")
    _write_submission(tmp_path, _valid_items())
    # Act
    result = build_gate_result(tmp_path, {})
    # Assert
    assert result["passed"] is False


def test_corrupt_public_metadata_has_attributable_assignment_error(tmp_path):
    # Corrupt public metadata is classified without exposing its contents.
    (tmp_path / "task.jsonl").write_text("{not json at all\n", encoding="utf-8")
    _write_submission(tmp_path, _valid_items())
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert kinds == ["invalid_assignment"]


@pytest.mark.parametrize("kind", list(_gate.FIX_HINTS))
def test_every_fix_hint_is_non_empty(kind):
    # Arrange
    hints = _gate.FIX_HINTS
    # Act
    hint = hints[kind]
    # Assert
    assert hint != ""


def test_check_id_is_dataset_submission_format():
    # Arrange
    module = _gate
    # Act
    value = module.CHECK_ID
    # Assert
    assert value == "dataset-submission-format"

# EOF


class TestTypedGateCallerContract:
    @staticmethod
    def task(workdir, *, answer_type=None, task_id=_TASK_ID_A, benchmark="corebench"):
        row = {"task_id": task_id}
        if benchmark is not None:
            row["benchmark"] = benchmark
        if answer_type is not None:
            row["answer_type"] = answer_type
        (workdir / "task.jsonl").write_text(json.dumps(row) + "\n")

    @pytest.mark.parametrize("declared,answer", [("number", 0), ("integer", -1), ("string", "0.94"), ("boolean", False), ("list", [0, None]), ("object", {"x": True})])
    def test_declared_public_type_valid_value_passes(self, tmp_path, declared, answer):
        self.task(tmp_path, answer_type=declared)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": answer}])
        assert build_gate_result(tmp_path, {})["passed"] is True

    @pytest.mark.parametrize("declared,answer", [("number", "0.94"), ("number", True), ("integer", 1.0), ("string", 1), ("boolean", 1), ("list", {"x": 0}), ("object", [0])])
    def test_declared_public_type_wrong_value_fails_without_coercion(self, tmp_path, declared, answer):
        self.task(tmp_path, answer_type=declared)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": answer}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        finding = next(f for f in result["findings"] if f["kind"] == "wrong_answer_type")
        assert finding["severity"] == "error"
        assert "$[0].answer" in finding["message"]
        assert _TASK_ID_A in finding["message"]
        assert declared in finding["fix_hint"]

    def test_typed_null_with_reason_preserves_honest_abstention(self, tmp_path):
        self.task(tmp_path, answer_type="number")
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": None, "reason": "agent abstained: unavailable source"}])
        assert build_gate_result(tmp_path, {})["passed"] is True

    def test_undeclared_json_and_extra_warning_remain_supported(self, tmp_path):
        self.task(tmp_path)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": {"x": [0, True]}, "confidence": 0.5}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is True
        assert next(f for f in result["findings"] if f["kind"] == "unknown_field")["severity"] == "warning"

    def test_duplicate_submission_cannot_hide_missing_task(self, tmp_path):
        _write_capsule(tmp_path)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}, {"task_id": _TASK_ID_A, "answer": 1}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        kinds = {f["kind"] for f in result["findings"]}
        assert {"duplicate_task_id", "missing_task_id"}.issubset(kinds)
        assert all(f["fix_hint"] for f in result["findings"])

    def test_unknown_submission_id_cannot_replace_selected_peer(self, tmp_path):
        self.task(tmp_path)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_B, "answer": 0}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert {"unknown_task_id", "missing_task_id"}.issubset({f["kind"] for f in result["findings"]})

    def test_multiple_capsule_candidates_are_refused_not_first_selected(self, tmp_path):
        for name in ["capsule-001", "capsule-002"]:
            child = tmp_path / name
            child.mkdir()
            self.task(child)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert [f["kind"] for f in result["findings"]] == ["ambiguous_capsule"]
        assert result["findings"][0]["fix_hint"]

    def test_direct_bound_task_scope_is_authoritative(self, tmp_path):
        self.task(tmp_path)
        child = tmp_path / "capsule-001"
        child.mkdir()
        self.task(child, task_id=_TASK_ID_B)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        assert build_gate_result(tmp_path, {})["passed"] is True

    def test_no_task_metadata_explicitly_reports_shape_only(self, tmp_path):
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {"benchmark": "corebench"})
        assert result["passed"] is True
        assert any(f["kind"] == "shape_only" and f["severity"] == "info" for f in result["findings"])

    def test_mixed_public_benchmark_declarations_refuse(self, tmp_path):
        rows = [{"task_id": _TASK_ID_A, "benchmark": "corebench"}, {"task_id": _TASK_ID_B, "benchmark": "bixbench"}]
        (tmp_path / "task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        _write_submission(tmp_path, _valid_items())
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert any(f["kind"] == "invalid_assignment" for f in result["findings"])

    def test_public_missing_benchmark_can_use_explicit_config(self, tmp_path):
        self.task(tmp_path, benchmark=None)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        assert build_gate_result(tmp_path, {"benchmark": "corebench"})["passed"] is True

    def test_missing_benchmark_without_config_does_not_guess(self, tmp_path):
        self.task(tmp_path, benchmark=None)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert any(f["kind"] == "invalid_assignment" for f in result["findings"])

    def test_public_invalid_benchmark_type_is_diagnosed(self, tmp_path):
        self.task(tmp_path, benchmark=1)
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert any(f["kind"] == "invalid_assignment" for f in result["findings"])

    def test_duplicate_public_task_rows_are_not_deduplicated(self, tmp_path):
        _write_capsule(tmp_path, task_ids=[_TASK_ID_A, _TASK_ID_A])
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert any(f["kind"] == "duplicate_assignment" for f in result["findings"])

    def test_feedback_and_dataclass_shape_never_echo_answer_payload(self, tmp_path):
        marker = "PRIVATE_ORACLE_SENTINEL_NOT_A_HINT"
        self.task(tmp_path, answer_type="number")
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": marker}])
        result = build_gate_result(tmp_path, {})
        assert result["passed"] is False
        assert marker not in json.dumps(result)
        for finding in result["findings"]:
            assert set(finding) == {"check_id", "kind", "message", "severity", "fix_hint"}

    def test_unexpected_exception_payload_is_sanitized_and_fail_closed(self, tmp_path):
        marker = "PRIVATE_EXCEPTION_PAYLOAD_NOT_A_HINT"
        class InvalidWorkdir:
            def __fspath__(self):
                raise RuntimeError(marker)
        result = build_gate_result(InvalidWorkdir(), {})
        assert result["passed"] is False
        assert result["findings"][0]["kind"] == "check_error"
        assert marker not in json.dumps(result)


class TestPublicBenchmarkMetadata:
    def test_same_declared_benchmark_is_carried_by_public_loader(self, tmp_path):
        from scitex_dataset.ai_for_science._validate import read_public_task_contract
        _write_capsule(tmp_path)
        contract = read_public_task_contract(tmp_path)
        assert contract["ok"] is True
        assert contract["benchmark"] == "corebench"

    def test_all_omitted_benchmarks_stay_unknown_in_public_loader(self, tmp_path):
        from scitex_dataset.ai_for_science._validate import read_public_task_contract
        rows = [{"task_id": _TASK_ID_A}, {"task_id": _TASK_ID_B}]
        (tmp_path / "task.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        contract = read_public_task_contract(tmp_path)
        assert contract["ok"] is True
        assert contract["benchmark"] is None

    def test_mixed_declared_and_omitted_benchmarks_are_not_guessed(self, tmp_path):
        from scitex_dataset.ai_for_science._validate import read_public_task_contract
        rows = [{"task_id": _TASK_ID_A, "benchmark": "corebench"}, {"task_id": _TASK_ID_B}]
        (tmp_path / "task.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        contract = read_public_task_contract(tmp_path)
        assert contract["ok"] is False
        assert any(e["kind"] == "invalid_assignment" and e["field"] == "benchmark" for e in contract["errors"])

    @pytest.mark.parametrize("benchmark", [None, "", "   ", 1])
    def test_explicit_malformed_public_benchmark_refuses(self, tmp_path, benchmark):
        from scitex_dataset.ai_for_science._validate import read_public_task_contract
        (tmp_path / "task.jsonl").write_text(json.dumps({"task_id": _TASK_ID_A, "benchmark": benchmark}) + "\n")
        assert read_public_task_contract(tmp_path)["ok"] is False

    @pytest.mark.parametrize("benchmark", [1, False, "   "])
    def test_bad_explicit_config_is_not_silently_treated_as_unknown(self, tmp_path, benchmark):
        _write_submission(tmp_path, [{"task_id": _TASK_ID_A, "answer": 0}])
        result = build_gate_result(tmp_path, {"benchmark": benchmark})
        assert result["passed"] is False
        assert result["findings"][0]["kind"] == "invalid_assignment"
