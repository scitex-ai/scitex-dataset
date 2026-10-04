#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/test__gate_plugin.py

"""Tests for the thin scitex-dev gate provider shim (``_gate_plugin``).

These SKIP in this repo's CI (``scitex_dev`` is not installed): the
importorskip below gates the whole module. When ``scitex_dev.gate`` IS
present, they assert the provider yields exactly one GateCheck whose
identity matches the contract and whose ``run`` agrees with the pure
:func:`build_gate_result`.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("scitex_dev.gate")

from scitex_dataset import _gate_plugin  # noqa: E402
from scitex_dataset.ai_for_science._gate import build_gate_result  # noqa: E402

_TASK_ID = "corebench/capsule-1111111__hard__q0"


def _bound_workdir(workdir: Path) -> Path:
    """Write a real 1-task capsule + a valid submission into ``workdir``."""
    (workdir / "task.jsonl").write_text(
        json.dumps({"benchmark": "corebench", "task_id": _TASK_ID}) + "\n",
        encoding="utf-8",
    )
    sub_dir = workdir / "submission"
    sub_dir.mkdir(exist_ok=True)
    (sub_dir / "submission.json").write_text(
        json.dumps([{"task_id": _TASK_ID, "answer": 0.81}]), encoding="utf-8"
    )
    return workdir


def test_provide_returns_single_check():
    # Arrange
    provider = _gate_plugin.provide
    # Act
    checks = provider()
    # Assert
    assert len(checks) == 1


def test_provide_check_has_contract_id():
    # Arrange
    check = _gate_plugin.provide()[0]
    # Act
    check_id = check.id
    # Assert
    assert check_id == "dataset-submission-format"


def test_provide_check_has_pre_submission_stage():
    # Arrange
    check = _gate_plugin.provide()[0]
    # Act
    stage = check.stage
    # Assert
    assert stage == "pre-submission"


def test_run_passed_matches_pure_logic(tmp_path):
    # Arrange
    _bound_workdir(tmp_path)
    check = _gate_plugin.provide()[0]
    # Act
    gate_result = check.run(tmp_path, {})
    # Assert
    assert gate_result.passed is build_gate_result(tmp_path, {})["passed"]


def test_run_findings_count_matches_pure_logic(tmp_path):
    # Arrange
    _bound_workdir(tmp_path)
    check = _gate_plugin.provide()[0]
    # Act
    gate_result = check.run(tmp_path, {})
    # Assert
    assert len(gate_result.findings) == len(build_gate_result(tmp_path, {})["findings"])

# EOF
