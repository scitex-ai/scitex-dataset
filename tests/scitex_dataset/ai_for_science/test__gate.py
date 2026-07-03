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


def test_fail_closed_finding_kind_is_check_error(tmp_path):
    # Arrange — same corrupt task.jsonl triggers the fail-closed guard.
    (tmp_path / "task.jsonl").write_text("{not json at all\n", encoding="utf-8")
    _write_submission(tmp_path, _valid_items())
    # Act
    kinds = [f["kind"] for f in build_gate_result(tmp_path, {})["findings"]]
    # Assert
    assert kinds == ["check_error"]


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
