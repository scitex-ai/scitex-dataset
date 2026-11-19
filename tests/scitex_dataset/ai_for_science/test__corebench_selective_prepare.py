#!/usr/bin/env python3
"""``corebench.prepare(capsule_ids=...)`` is self-contained: it bootstraps the
oracle manifests standardize needs when (and only when) they are not staged.

Module seams are swapped and restored by hand (no unittest.mock): ``download``
reports success, and the first of ``bootstrap_oracle`` / ``standardize`` that
runs records itself and stops the pipeline.
"""

from contextlib import contextmanager

import pytest

from scitex_dataset.ai_for_science import corebench
from scitex_dataset.ai_for_science._base import resolve_paths
from scitex_dataset.ai_for_science._corebench_download import oracle_staged


class _Stop(Exception):
    pass


@contextmanager
def _seams(calls):
    names = ("download", "bootstrap_oracle", "standardize")
    saved = {n: getattr(corebench, n) for n in names}

    def _download(**_):
        return {"n_failed": 0, "failures": []}

    def _stage(name):
        def run(*_, **__):
            calls.append(name)
            raise _Stop

        return run

    corebench.download = _download
    corebench.bootstrap_oracle = _stage("bootstrap_oracle")
    corebench.standardize = _stage("standardize")
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(corebench, n, f)


def _stage_oracle(raw):
    (raw / "dataset").mkdir(parents=True, exist_ok=True)
    (raw / "dataset" / "core_train.json").write_text("[]")
    (raw / "core_test.json").write_text("[]")


def _first_stage(tmp_path, *, staged, capsule_ids):
    raw = resolve_paths("corebench", dataset_root=tmp_path).raw_dir
    raw.mkdir(parents=True)
    if staged:
        _stage_oracle(raw)
    calls: list[str] = []
    with _seams(calls), pytest.raises(_Stop):
        corebench.prepare(
            dataset_root=tmp_path, capsule_ids=capsule_ids, skip_inventory=True
        )
    return calls[0]


def test_selective_prepare_bootstraps_a_missing_oracle(tmp_path):
    # Arrange
    ids = ["capsule-1111111"]
    # Act
    first = _first_stage(tmp_path, staged=False, capsule_ids=ids)
    # Assert
    assert first == "bootstrap_oracle"


def test_selective_prepare_reuses_a_staged_oracle(tmp_path):
    # Arrange
    ids = ["capsule-1111111"]
    # Act
    first = _first_stage(tmp_path, staged=True, capsule_ids=ids)
    # Assert
    assert first == "standardize"


def test_oracle_staged_needs_both_manifests(tmp_path):
    # Arrange
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "core_train.json").write_text("[]")
    # Act
    staged = oracle_staged(tmp_path)
    # Assert
    assert staged is False
