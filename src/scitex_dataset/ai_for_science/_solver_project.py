#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_solver_project.py

"""Solver-ready PROJECT from one prepared capsule — and its answer guard.

``standardize`` writes one ``for_solver/capsule-NNN/`` dir per capsule
(``input/`` + ``task.jsonl`` + submission schema/example + README). Those
task rows still carry the dataset's task ids and the benchmark name, so a
harness that hands a capsule to an agent has to assemble a NEUTRAL project
from it. This module is that assembly, owned here so every harness gets
the same contract instead of re-implementing it:

* :func:`build_solver_project` — ``input/`` (minus OS junk) copied to
  ``out`` plus ``questions.json`` ``[{"id": "q1", "question": ...}]``.
  The ``q<n> -> task_id`` mapping is written to a SEALED path outside the
  project (operator side, like ``eval/``). The finished project is then
  checked by :func:`assert_solver_project_clean`; on any violation the
  project and the mapping are removed and :class:`SolverProjectLeakError`
  is raised, so an unclean project never exists on disk.
* :func:`prepare_solver_project` — the same, addressed by benchmark +
  capsule (friendly ``capsule-NNN`` or native id) under a dataset root.
* :func:`submission_from_answers` — the agent's ``{"q1": value, ...}``
  back to the uniform ``[{"task_id", "answer"}]`` submission.

The guard is dataset-blind. A solver project must contain NONE of:

1. an operator-side file NAME (``answers.jsonl``, ``evaluate.py``,
   ``path_renames.json``, ``sources.jsonl``, ``index.jsonl``, the
   materialization ledger, ``MANIFEST.yaml``, ``inventory.json``, or an
   upstream answer manifest a benchmark declares);
2. a file byte-identical to any file under ``eval/`` (a renamed copy of
   the answer key is still the answer key);
3. a symlink resolving outside the project (it could reach ``raw/`` or
   ``eval/``);
4. a task id, its local part, or a native capsule identifier — in a path
   name or in readable text — or a path name matching the benchmark
   adapter's ``PATH_ID_PATTERNS`` (e.g. a capsule-level ``bix-<n>``);
5. an answer value of the capsule's own tasks, by the SAME precision-aware
   matcher the ``standardize`` masking uses (:func:`_assert_no_value_leak`).

Like that matcher, the guard is bounded (readable text up to 5 MB); a pass
is evidence, not proof, that no answer is reachable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from ._base import resolve_paths
from ._capsule_layout import (
    _IDENTITY_FILENAME,
    _INDEX_FILENAME,
    _INPUT_SUBDIR,
    _TASK_FILENAME,
    _resolve_only,
)
from ._identifiers import native_identifiers
from ._leak_strip import AnswerLeakError, _assert_no_value_leak, _is_value_scan_target

QUESTIONS_FILE = "questions.json"
ANSWERS_FILE = "answers.json"

#: Operator-side file names that must never appear in a solver project.
OPERATOR_FILE_NAMES = frozenset(
    {
        "answers.jsonl",
        "evaluate.py",
        "path_renames.json",
        "sources.jsonl",
        "inventory.json",
        "MANIFEST.yaml",
        _INDEX_FILENAME,
        _IDENTITY_FILENAME,
        _TASK_FILENAME,
    }
)

#: Upstream answer-bearing manifests, per benchmark (kept in ``raw/``).
UPSTREAM_ORACLE_NAMES = frozenset(
    {"core_test.json", "core_train.json", "BixBench.jsonl"}
)

_JUNK = shutil.ignore_patterns(".DS_Store", "__MACOSX", "._*")
_TEXT_LIMIT = 5_000_000


class SolverProjectLeakError(RuntimeError):
    """A solver project holds operator material; ``violations`` lists it.

    Messages name paths and violation kinds only — never an answer value.
    """

    def __init__(self, violations: list[dict]):
        self.violations = violations
        head = "; ".join(f"{v['kind']}: {v['path']}" for v in violations[:5])
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        super().__init__(f"solver project is not clean: {head}{more}")


# ---------------------------------------------------------------------------
# Reading the operator side
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _leaves(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def eval_answer_values(eval_dir: Path | str, task_ids) -> list[object]:
    """Every reference value ``eval/answers.jsonl`` holds for ``task_ids``.

    Benchmark-agnostic: all scalar leaves of each row's ``answer`` (every
    reference run of a task included); a string that parses as a finite
    number is also passed as that number. Booleans/short strings/integers
    are left to the shared target policy (they are never targets).
    """
    wanted = set(task_ids)
    path = Path(eval_dir) / "answers.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"eval answer key not found: {path}")
    values: list[object] = []
    for row in _read_jsonl(path):
        if row.get("task_id") not in wanted:
            continue
        for leaf in _leaves(row.get("answer")):
            if leaf is None or isinstance(leaf, bool):
                continue
            values.append(leaf)
            if isinstance(leaf, str):
                try:
                    number = float(leaf.strip())
                except ValueError:
                    continue
                if math.isfinite(number):
                    values.append(number)
    return values


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_digests(eval_dir: Path) -> set[str]:
    return {
        _sha256(p)
        for p in sorted(Path(eval_dir).rglob("*"))
        if p.is_file() and not p.is_symlink() and p.stat().st_size > 0
    }


def _task_identifiers(task_ids, native_id: str | None) -> list[str]:
    found = set(native_identifiers(native_id))
    for tid in task_ids:
        found.add(tid)
        local = tid.split("/", 1)[-1]
        if local:
            found.add(local)
    return sorted(found, key=lambda s: (-len(s), s))


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def assert_solver_project_clean(
    project: Path | str,
    *,
    eval_dir: Path | str,
    task_ids,
    native_id: str | None = None,
    path_patterns=(),
    extra_operator_names=(),
) -> dict:
    """Refuse a solver project that holds any operator material.

    See the module docstring for the five checks. ``path_patterns`` are the
    benchmark adapter's id regexes (``PATH_ID_PATTERNS``, the same ones
    ``standardize`` enforces on path names); :func:`build_solver_project`
    supplies them from the task rows' ``benchmark``. Returns a small report
    when clean; raises :class:`SolverProjectLeakError` listing every
    violation otherwise.
    """
    project = Path(project)
    eval_dir = Path(eval_dir)
    task_ids = list(task_ids)
    names = OPERATOR_FILE_NAMES | UPSTREAM_ORACLE_NAMES | set(extra_operator_names)
    digests = _eval_digests(eval_dir)
    idents = _task_identifiers(task_ids, native_id)
    patterns = [re.compile(p) for p in path_patterns]
    top = project.resolve()
    violations: list[dict] = []
    n_files = 0

    for p in sorted(project.rglob("*")):
        rel = str(p.relative_to(project))
        if p.name in names:
            violations.append({"kind": "operator-file", "path": rel})
        if any(i in rel for i in idents) or any(r.search(rel) for r in patterns):
            violations.append({"kind": "identifier-in-path", "path": rel})
        if p.is_symlink():
            target = Path(os.path.realpath(p))
            if not target.is_relative_to(top):
                violations.append({"kind": "symlink-outside", "path": rel})
            continue
        if not p.is_file():
            continue
        n_files += 1
        if digests and p.stat().st_size > 0 and _sha256(p) in digests:
            violations.append({"kind": "eval-copy", "path": rel})
        if p.stat().st_size <= _TEXT_LIMIT and _is_value_scan_target(p):
            try:
                text = p.read_text(errors="ignore")
            except OSError:  # pragma: no cover - read race
                continue
            if any(i in text for i in idents):
                violations.append({"kind": "identifier-in-text", "path": rel})

    values = eval_answer_values(eval_dir, task_ids)
    try:
        _assert_no_value_leak(project, values, capsule=native_id or "<project>")
    except AnswerLeakError as exc:
        # The message carries a path and a char offset, never the value.
        violations.append({"kind": "answer-value", "path": str(exc)})

    if violations:
        raise SolverProjectLeakError(violations)
    return {
        "clean": True,
        "n_files": n_files,
        "n_task_ids": len(task_ids),
        "n_reference_values": len(values),
        "n_eval_digests": len(digests),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _native_id_of(task_dir: Path) -> str | None:
    index = task_dir.parent / _INDEX_FILENAME
    if not index.is_file():
        return None
    for row in _read_jsonl(index):
        if row.get("dir") == task_dir.name:
            return row.get("native_id")
    return None


def _adapter_path_patterns(tasks: list[dict]) -> tuple[str, ...]:
    """The benchmark adapter's ``PATH_ID_PATTERNS`` (empty if it has none)."""
    benches = {t.get("benchmark") for t in tasks if t.get("benchmark")}
    patterns: list[str] = []
    for bench in sorted(benches):
        try:
            module = importlib.import_module(f".{bench}", package=__package__)
        except ModuleNotFoundError:
            continue
        patterns.extend(getattr(module, "PATH_ID_PATTERNS", ()))
    return tuple(patterns)


def _refuse_overlap(out: Path, mapping: Path, *operator_dirs: Path) -> None:
    """The mapping must lie outside the project; operator dirs (``eval/``,
    the capsule dir) and the project must not contain one another."""
    o = out.resolve()
    if mapping.resolve().is_relative_to(o):
        raise ValueError(f"the mapping {mapping} must live outside the project {out}")
    for p in operator_dirs:
        r = p.resolve()
        if r.is_relative_to(o) or o.is_relative_to(r):
            raise ValueError(
                f"operator dir {p} and solver project {out} must not contain each other"
            )


def build_solver_project(
    task_dir: Path | str,
    out: Path | str,
    *,
    mapping: Path | str,
    eval_dir: Path | str,
) -> dict:
    """Assemble and guard a neutral solver project from one capsule dir.

    ``task_dir`` is a prepared ``for_solver/capsule-NNN/``; ``out`` must not
    exist; ``mapping`` (the ``q<n> -> task_id`` map) and ``eval_dir`` must
    lie outside ``out``. Returns the mapping document plus the guard report.
    """
    task_dir, out, mapping, eval_dir = (
        Path(task_dir),
        Path(out),
        Path(mapping),
        Path(eval_dir),
    )
    if out.exists():
        raise FileExistsError(out)
    _refuse_overlap(out, mapping, eval_dir, task_dir)
    tasks = _read_jsonl(task_dir / _TASK_FILENAME)
    native_id = _native_id_of(task_dir)

    shutil.copytree(task_dir / _INPUT_SUBDIR, out, ignore=_JUNK, symlinks=True)
    questions = [
        {"id": f"q{i}", "question": t["prompt"]} for i, t in enumerate(tasks, 1)
    ]
    (out / QUESTIONS_FILE).write_text(
        json.dumps(questions, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    doc = {
        "task_dir": str(task_dir),
        "project": str(out),
        "native_id": native_id,
        "ids": {q["id"]: t["task_id"] for q, t in zip(questions, tasks)},
    }
    try:
        doc["guard"] = assert_solver_project_clean(
            out,
            eval_dir=eval_dir,
            task_ids=doc["ids"].values(),
            native_id=native_id,
            path_patterns=_adapter_path_patterns(tasks),
        )
    except SolverProjectLeakError:
        shutil.rmtree(out, ignore_errors=True)
        raise
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def prepare_solver_project(
    benchmark: str,
    capsule: str,
    out: Path | str,
    *,
    mapping: Path | str,
    dataset_root: Path | str | None = None,
    dry_run: bool = False,
) -> dict:
    """:func:`build_solver_project` addressed by benchmark + capsule id.

    ``capsule`` is a friendly ``capsule-NNN`` or a native capsule id; the
    capsule must already be materialized by ``prepare``/``standardize``.
    ``dry_run`` resolves the capsule and returns the plan, writing nothing.
    """
    paths = resolve_paths(benchmark, dataset_root=dataset_root)
    index = _read_jsonl(paths.for_solver_dir / _INDEX_FILENAME)
    row = _resolve_only(capsule, index)
    task_dir = paths.for_solver_dir / row["dir"]
    if not (task_dir / _INPUT_SUBDIR).is_dir():
        raise FileNotFoundError(
            f"{benchmark} {capsule}: not materialized at {task_dir}; run "
            f"prepare --only {capsule} first"
        )
    if dry_run:
        return {
            "dry_run": True,
            "task_dir": str(task_dir),
            "project": str(out),
            "mapping": str(mapping),
            "n_questions": len(row["task_ids"]),
        }
    return build_solver_project(task_dir, out, mapping=mapping, eval_dir=paths.eval_dir)


def submission_from_answers(
    answers: Path | str | None, mapping: Path | str
) -> list[dict]:
    """The uniform ``[{"task_id", "answer"}]`` for every mapped question.

    An unanswered question (or a missing/unreadable/non-object answers
    file) becomes ``answer: None`` so it is scored as a failure, not
    dropped.
    """
    ids = json.loads(Path(mapping).read_text(encoding="utf-8"))["ids"]
    given: dict = {}
    if answers is not None and Path(answers).is_file():
        try:
            given = json.loads(Path(answers).read_text(encoding="utf-8"))
        except ValueError:
            given = {}
    if not isinstance(given, dict):
        given = {}
    return [{"task_id": tid, "answer": given.get(qid)} for qid, tid in ids.items()]


__all__ = [
    "ANSWERS_FILE",
    "OPERATOR_FILE_NAMES",
    "QUESTIONS_FILE",
    "UPSTREAM_ORACLE_NAMES",
    "SolverProjectLeakError",
    "assert_solver_project_clean",
    "build_solver_project",
    "eval_answer_values",
    "prepare_solver_project",
    "submission_from_answers",
]

# EOF
