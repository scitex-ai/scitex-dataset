#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_oracle.py

"""Operator oracle (``eval/answers.jsonl``) loading (split from ``_score``).

HOST-SIDE ONLY. Diagnostics carry line numbers, kinds and field NAMES, never
reference values. Re-exported from :mod:`._score`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ._reference import reference_issue


class OracleIntegrityError(ValueError):
    """Strict oracle parsing refused; diagnostics contain no reference values."""

    def __init__(self, diagnostics: list[dict]):
        self.diagnostics = diagnostics
        kinds = Counter(d.get("kind") for d in diagnostics)
        summary = ", ".join(f"{k}: {n}" for k, n in sorted(kinds.items()))
        fields = sorted({d["field"] for d in diagnostics if "field" in d})
        detail = f" (reference field: {', '.join(fields)})" if fields else ""
        super().__init__(
            f"invalid oracle records [{summary}]{detail}; inspect line/kind diagnostics"
        )


def _load_oracle(
    answers: Path | str,
    *,
    diagnostics: list[dict] | None = None,
    strict: bool = False,
    reference_field: str | None = None,
) -> dict[str, list[Any]]:
    """Load ``answers.jsonl`` into ``{full_task_id: [answer_payload, ...]}``.

    ``answers`` may be the ``answers.jsonl`` file itself or a directory
    holding it (the eval dir). Records are grouped by FULL ``task_id``
    preserving first-seen order, so a task_id repeated across N oracle
    lines yields N reference values for the numeric prediction interval.
    Legacy tolerant parsing remains the default. Qualification callers may
    request ``strict=True``; otherwise discarded-record diagnostics can be
    collected without copying private answer values. A required ``answer``
    field must be present; present null remains an explicit reference value.
    With an adapter ``reference_field``, a dict payload lacking that key is
    refused (``missing_reference_field``), never graded against some other
    key (#115). A present-but-null/blank reference is kept, like any invalid
    reference, and attributed per task as ``invalid_reference``.
    """
    p = Path(answers)
    if p.is_dir():
        p = p / "answers.jsonl"
    grouped: dict[str, list[Any]] = {}
    issues: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                issues.append({"line": line_number, "kind": "unparseable_record"})
                continue
            if not isinstance(obj, dict):
                issues.append({"line": line_number, "kind": "record_not_object"})
                continue
            tid = obj.get("task_id")
            if not isinstance(tid, str) or not tid.strip():
                issues.append({"line": line_number, "kind": "invalid_task_id"})
                continue
            if "answer" not in obj:
                issues.append({"line": line_number, "kind": "missing_answer_field"})
                continue
            kind = reference_issue(obj["answer"], reference_field)
            if kind == "missing_reference_field":
                issues.append({"line": line_number, "kind": kind, "field": reference_field})
                continue
            grouped.setdefault(tid, []).append(obj["answer"])
    if diagnostics is not None:
        diagnostics.extend(issues)
    if strict and issues:
        raise OracleIntegrityError(issues)
    return grouped


__all__ = ["OracleIntegrityError", "_load_oracle"]

# EOF
