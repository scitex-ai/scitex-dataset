#!/usr/bin/env python3
"""Eligibility config validation and the predicate evaluator.

These are pure functions over a config and a fact record; the fact records
here are plain dicts shaped like what the adapters return (no I/O stands in
for anything — the real-data path is covered in
``tests/integration/ai_for_science/test_eligibility_real.py``).
"""

import pytest

from scitex_dataset.ai_for_science._eligibility._config import (
    EligibilityConfigError,
    config_sha256,
    evaluate,
    load_config,
)


def _cfg(**over):
    base = {"benchmark": "corebench",
            "rules": [{"id": "r1", "fact": "n_questions", "min": 3}]}  # fmt: skip
    base.update(over)
    return base


def test_rule_on_archive_fact_is_staged_archive():
    # Arrange
    cfg = _cfg(rules=[{"id": "a", "fact": "dl_frameworks", "none_of": ["torch"]},
                      {"id": "m", "fact": "split", "in": ["train"]}])  # fmt: skip
    # Act
    stages = [r["stage"] for r in load_config(cfg)["rules"]]
    # Assert
    assert stages == ["archive", "metadata"]


def test_normalized_config_loads_unchanged():
    # Arrange
    once = load_config(_cfg())
    # Act
    twice = load_config(once)
    # Assert
    assert config_sha256(twice) == config_sha256(once)


@pytest.mark.parametrize(
    "bad",
    [
        {"rules": [{"id": "x", "fact": "no_such_fact", "equals": 1}]},
        {"rules": [{"id": "x", "fact": "split", "like": "train"}]},
        {"rules": [{"id": "x", "fact": "split", "in": "train"}]},
        {"rules": [{"id": "x", "fact": "n_questions", "min": True}]},
        {
            "rules": [
                {"id": "x", "fact": "split", "in": ["a"]},
                {"id": "x", "fact": "split", "in": ["b"]},
            ]
        },  # fmt: skip
        {"rules": [{"id": "x", "all": []}]},
        {"order": {"keys": [{"fact": "dl_frameworks"}]}},
        {"target": 0},
        {"on_unknown": "ignore"},
        {"typo_key": 1},
    ],
)
def test_malformed_config_is_refused(bad):
    # Arrange
    cfg = _cfg(**bad)
    # Act
    try:
        load_config(cfg)
        error = None
    except EligibilityConfigError as exc:
        error = exc
    # Assert
    assert isinstance(error, EligibilityConfigError)


def test_unknown_fact_value_is_unknown_not_false():
    # Arrange
    pred = {"fact": "archive_bytes", "max": 10}
    # Act
    ok, checks = evaluate(pred, {"archive_bytes": None})
    # Assert
    assert ok is None and checks[0]["value"] is None


def test_all_is_false_when_one_part_fails_even_if_another_is_unknown():
    # Arrange
    pred = {"all": [{"fact": "split", "in": ["train"]},
                    {"fact": "archive_bytes", "max": 10}]}  # fmt: skip
    # Act
    ok, _ = evaluate(pred, {"split": "test", "archive_bytes": None})
    # Assert
    assert ok is False


def test_any_holds_when_one_part_holds():
    # Arrange
    pred = {"any": [{"fact": "n_entry_steps", "min": 2},
                    {"fact": "entry_notebook_code_cells", "min": 3}]}  # fmt: skip
    # Act
    ok, _ = evaluate(pred, {"n_entry_steps": 1, "entry_notebook_code_cells": 36})
    # Assert
    assert ok is True


@pytest.mark.parametrize(
    "op,expected,value,ok",
    [
        ("none_of", ["torch", "jax"], ["keras", "torch"], False),
        ("any_of", ["python", "notebook"], ["notebook"], True),
        ("subset_of", ["python"], [], True),
        ("subset_of", ["python"], ["python", "r"], False),
        ("not_in", ["capsule-1"], "capsule-2", True),
    ],
)
def test_list_and_membership_operators(op, expected, value, ok):
    # Arrange
    pred = {"fact": "code_languages", op: expected}
    # Act
    result, _ = evaluate(pred, {"code_languages": value})
    # Assert
    assert result is ok


def test_extra_fact_reaches_adapter_metadata():
    # Arrange
    pred = {"fact": "extra.human_solvable", "in": ["yes"]}
    # Act
    ok, _ = evaluate(pred, {"extra": {"human_solvable": "no"}})
    # Assert
    assert ok is False
