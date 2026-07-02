#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_validate.py

"""Structural, oracle-free submission validation (HOST-SIDE).

:func:`validate_submission` checks an agent submission against the
UNIFORM submission contract (:data:`._standardize.UNIFORM_SUBMISSION_SCHEMA`)
and returns STRUCTURED errors — never a bare pass/fail::

    {"ok": bool, "errors": [{"path", "kind", "message"}, ...]}

**Oracle-free by construction.** This module imports NOTHING from the
scorer (:mod:`._score`) or any oracle loader — it is impossible for it
to read ``eval/answers.jsonl``. It imports only the uniform schema
constant from :mod:`._standardize` (a writer/schema module, not an
oracle reader). Validation therefore runs with no oracle present, which
is exactly what a solver-side / pre-flight check needs.

Checks performed (per the operator-locked spec):

- top level must be a JSON array (``wrong_type`` otherwise);
- each item must be an object with a string ``task_id`` and a present
  ``answer`` field (``missing_field`` / ``wrong_type``);
- ``task_id`` must carry the benchmark prefix and, for corebench, the
  ``<native>__<difficulty>__q<N>`` shape (``bad_task_id``);
- array length vs the expected task count, when ``expected_task_ids``
  is supplied (``wrong_count``);
- any key beyond ``task_id`` / ``answer`` / ``reason`` is reported as a
  non-fatal WARNING (``unknown_field``).

Error ``kind`` values in :data:`WARN_KINDS` are warnings and do NOT
flip ``ok`` to False; every other kind is a hard error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# NOTE: the ONLY sibling import is the schema constant — a leak-safe
# writer/schema module. NEVER import ._score or any oracle loader here.
from ._standardize import UNIFORM_SUBMISSION_SCHEMA

# Keys a submission row may legitimately carry. ``reason`` is used by the
# abstention protocol; anything else is surfaced as an unknown-field WARN.
_KNOWN_ITEM_KEYS = {"task_id", "answer", "reason"}

# Error kinds that are WARNINGS — reported but do not flip ``ok`` False.
WARN_KINDS = {"unknown_field"}


def _err(path: str, kind: str, message: str) -> dict:
    return {"path": path, "kind": kind, "message": message}


def _coerce(submission: Any) -> tuple[Any, dict | None]:
    """Return ``(parsed, load_error)`` for a path/str or already-parsed value.

    A path that does not exist → ``no_file``; a path whose contents are
    not JSON → ``unparseable``. An already-parsed object is returned
    unchanged with no load error.
    """
    if isinstance(submission, (str, Path)):
        p = Path(submission)
        if not p.exists():
            return None, _err("$", "no_file", f"submission file not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, _err("$", "unparseable", f"submission is not valid JSON: {exc}")
    return submission, None


def _valid_task_id(benchmark: str, task_id: str) -> bool:
    """True iff ``task_id`` carries the benchmark prefix and expected shape.

    All benchmarks require a ``<benchmark>/<rest>`` prefix with a
    non-empty ``rest``. CoreBench additionally requires the
    ``<native>__<difficulty>__q<N>`` triple its standardizer emits.
    """
    prefix = f"{benchmark}/"
    if not task_id.startswith(prefix):
        return False
    rest = task_id[len(prefix):]
    if not rest:
        return False
    if benchmark == "corebench":
        parts = rest.split("__")
        if len(parts) != 3:
            return False
        native, difficulty, q = parts
        if not native or not difficulty:
            return False
        if not (q.startswith("q") and q[1:].isdigit()):
            return False
    return True


def expected_task_ids_from_for_solver(for_solver_dir: Path | str) -> list[str] | None:
    """Collect the expected task_ids from a for_solver ``index.jsonl`` mapper.

    Reads only the AGENT-VISIBLE ``for_solver/index.jsonl`` (never the
    oracle ``eval/answers.jsonl``), flattening every capsule row's
    ``task_ids``. Returns ``None`` when no index is present, so the
    caller can skip the count check rather than fail.
    """
    idx = Path(for_solver_dir) / "index.jsonl"
    if not idx.is_file():
        return None
    task_ids: list[str] = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for tid in row.get("task_ids", []) or []:
            if isinstance(tid, str):
                task_ids.append(tid)
    return task_ids or None


def validate_submission(
    benchmark: str,
    submission: Any,
    *,
    expected_task_ids: list[str] | None = None,
) -> dict:
    """Structurally validate ``submission`` against the uniform contract.

    ``submission`` may be a path (``str`` / ``Path``) to a JSON file or
    an already-parsed Python object. ``expected_task_ids`` (optional,
    agent-visible) enables the array-length check.

    Returns ``{"ok": bool, "errors": [{"path", "kind", "message"}]}``.
    ``ok`` is True iff no error whose ``kind`` is outside
    :data:`WARN_KINDS` was recorded.
    """
    errors: list[dict] = []
    data, load_err = _coerce(submission)
    if load_err is not None:
        return {"ok": False, "errors": [load_err]}

    if not isinstance(data, list):
        errors.append(
            _err("$", "wrong_type", "submission must be a JSON array of objects")
        )
        return {"ok": False, "errors": errors}

    for i, item in enumerate(data):
        path = f"$[{i}]"
        if not isinstance(item, dict):
            errors.append(_err(path, "wrong_type", "submission item must be an object"))
            continue

        if "task_id" not in item:
            errors.append(_err(path, "missing_field", "item is missing 'task_id'"))
        else:
            tid = item["task_id"]
            if not isinstance(tid, str):
                errors.append(
                    _err(f"{path}.task_id", "wrong_type", "'task_id' must be a string")
                )
            elif not _valid_task_id(benchmark, tid):
                errors.append(
                    _err(
                        f"{path}.task_id",
                        "bad_task_id",
                        f"'task_id' {tid!r} does not match the {benchmark} shape",
                    )
                )

        if "answer" not in item:
            errors.append(_err(path, "missing_field", "item is missing 'answer'"))

        for key in item:
            if key not in _KNOWN_ITEM_KEYS:
                errors.append(
                    _err(
                        f"{path}.{key}",
                        "unknown_field",
                        f"unknown field {key!r} (not in the uniform schema)",
                    )
                )

    if expected_task_ids is not None and len(data) != len(expected_task_ids):
        errors.append(
            _err(
                "$",
                "wrong_count",
                f"submission has {len(data)} item(s); expected "
                f"{len(expected_task_ids)}",
            )
        )

    ok = not any(e["kind"] not in WARN_KINDS for e in errors)
    return {"ok": ok, "errors": errors}


# Referenced so the uniform schema stays coupled to the validator (the
# required-field / task_id / answer checks above mirror its structure).
_SCHEMA = UNIFORM_SUBMISSION_SCHEMA


__all__ = [
    "validate_submission",
    "expected_task_ids_from_for_solver",
    "WARN_KINDS",
]

# EOF
