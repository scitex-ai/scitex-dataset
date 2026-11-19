#!/usr/bin/env python3
"""Candidate ordering of the screening engine (sort keys, nulls, seeded draw)."""

from scitex_dataset.ai_for_science._eligibility._engine import _order


def _units():
    return [{"unit_id": f"u-{i}", "n_questions": q, "archive_bytes": b}
            for i, (q, b) in enumerate([(2, 5), (3, 1), (3, 9), (2, None), (1, 7)])]  # fmt: skip


def _ids(units, keys=(), seed=None):
    return [u["unit_id"] for u in _order(units, {"keys": list(keys), "seed": seed})]


def test_order_keys_descending_with_nulls_last_and_natural_id_tiebreak():
    # Arrange
    keys = [{"fact": "n_questions", "descending": True},
            {"fact": "archive_bytes", "descending": True}]  # fmt: skip
    # Act
    ids = _ids(_units(), keys)
    # Assert
    assert ids == ["u-2", "u-1", "u-0", "u-3", "u-4"]


def test_natural_order_puts_bix_6_before_bix_10():
    # Arrange
    units = [{"unit_id": "bix-10"}, {"unit_id": "bix-6"}, {"unit_id": "bix-51"}]
    # Act
    ids = _ids(units)
    # Assert
    assert ids == ["bix-6", "bix-10", "bix-51"]


def test_seeded_order_does_not_depend_on_input_order():
    # Arrange
    units = _units()
    # Act
    reordered = _ids(list(reversed(units)), seed=7)
    # Assert
    assert reordered == _ids(units, seed=7)


def test_seeded_order_changes_with_the_seed():
    # Arrange
    reference = _ids(_units(), seed=7)
    # Act
    others = [_ids(_units(), seed=s) for s in range(8, 20)]
    # Assert
    assert any(o != reference for o in others)
