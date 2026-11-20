"""Guarded solver projects over REAL prepared capsules (opt-in).

Set ``SCITEX_DATASET_REAL_ROOT`` to a dataset root that already holds
``ai-for-science/<benchmark>/{for_solver,eval}`` prepared from the real
upstream data (CI has no such download, so the module is skipped there).
Every materialized capsule of every benchmark is assembled into a solver
project and must pass the guard; a copy with the eval answer key planted
inside must be refused.
"""

import os
import shutil
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science._solver_project import (
    SolverProjectLeakError,
    assert_solver_project_clean,
    build_solver_project,
)

ROOT = os.environ.get("SCITEX_DATASET_REAL_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="SCITEX_DATASET_REAL_ROOT not set")


def _capsules():
    if not ROOT:
        return []
    found = []
    for bench in sorted((Path(ROOT) / "ai-for-science").glob("*")):
        for d in sorted((bench / "for_solver").glob("capsule-*")):
            if (d / "input").is_dir():
                found.append(pytest.param(d, id=f"{bench.name}-{d.name}"))
    return found


@pytest.fixture(params=_capsules())
def built(request, tmp_path):
    task_dir = request.param
    eval_dir = task_dir.parent.parent / "eval"
    doc = build_solver_project(task_dir, tmp_path / "project", mapping=tmp_path / "sealed.json",
                               eval_dir=eval_dir)  # fmt: skip
    return {"doc": doc, "project": tmp_path / "project", "eval": eval_dir}


def test_real_capsule_project_passes_the_guard(built):
    # Arrange
    doc = built["doc"]
    # Act
    clean = doc["guard"]["clean"]
    # Assert
    assert clean is True


def test_real_capsule_project_with_planted_answer_key_is_refused(built):
    # Arrange
    shutil.copy(built["eval"] / "answers.jsonl", built["project"] / "notes.dat")
    # Act
    try:
        assert_solver_project_clean(built["project"], eval_dir=built["eval"],
                                    task_ids=built["doc"]["ids"].values(),
                                    native_id=built["doc"]["native_id"])  # fmt: skip
        kinds = set()
    except SolverProjectLeakError as exc:
        kinds = {v["kind"] for v in exc.violations}
    # Assert
    assert "eval-copy" in kinds


def test_real_capsule_mapping_lists_every_task_row(built):
    # Arrange
    task_rows = (Path(built["doc"]["task_dir"]) / "task.jsonl").read_text().splitlines()
    # Act
    n_ids = len(built["doc"]["ids"])
    # Assert
    assert n_ids == len([r for r in task_rows if r.strip()])
