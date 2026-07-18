#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_gate.py

"""Pure, scitex_dev-AGNOSTIC logic for the submission-format gate check.

This module implements the whole decision of the scitex-dev
``pre-submission`` gate ``dataset-submission-format`` WITHOUT importing
``scitex_dev``. It imports only scitex-dataset internals (the shipped
:func:`._validate.validate_submission`) and returns a plain dict that the
thin plugin shim (:mod:`scitex_dataset._gate_plugin`) maps onto the
frozen ``scitex_dev.gate`` dataclasses.

The check is structural and ORACLE-FREE — it never reads
``eval/answers.jsonl``; it only validates the *shape* of a submission
against the bound capsule's ``task.jsonl``. Its ``build_gate_result`` is
FAIL-CLOSED: any unexpected exception is caught and reported as a
``check_error`` finding rather than propagated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ._validate import WARN_KINDS, validate_submission

CHECK_ID = "dataset-submission-format"

# Format-specific remediation hints, keyed by the ``kind`` values that
# ``validate_submission`` emits. NOT clew's @stx.session hint — these are
# purely about the JSON submission-array contract.
FIX_HINTS: dict[str, str] = {
    "no_file": (
        "Write your answers as a JSON array to `submission/submission.json` "
        "in the capsule workdir (see submission.example.json)."
    ),
    "unparseable": (
        "The submission file is not valid JSON — it must be a JSON array "
        'of {"task_id", "answer"} objects.'
    ),
    "wrong_type": (
        "Each item must be an object; the submission must be a JSON array "
        "of objects."
    ),
    "missing_field": "Each item needs both 'task_id' and 'answer'.",
    "missing_reason": (
        "A null answer must carry a non-empty 'reason' naming the proximal "
        "cause (honest abstention) — add a one-line reason, or provide an "
        "answer."
    ),
    "bad_task_id": (
        "Use the exact task_id(s) from this capsule's task.jsonl / "
        "submission.example.json."
    ),
    "unknown_field": "Remove fields other than 'task_id', 'answer' (and 'reason').",
    "wrong_count": (
        "Submit exactly one item per task_id in this capsule (see task.jsonl)."
    ),
}


def _find_capsule_dir(workdir: Path) -> Path | None:
    """Return the bound capsule dir (has a ``task.jsonl``), else ``None``.

    Prefers ``workdir`` itself, then the ``capsule-*`` subdirs in sorted
    order — the first one carrying a ``task.jsonl`` wins.
    """
    for candidate in [workdir, *sorted(workdir.glob("capsule-*"))]:
        if (candidate / "task.jsonl").is_file():
            return candidate
    return None


def _read_task_rows(capsule_dir: Path) -> list[dict]:
    """Parse the capsule's ``task.jsonl`` (one JSON object per line)."""
    rows: list[dict] = []
    text = (capsule_dir / "task.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_gate_result(workdir: Any, config: Mapping | None) -> dict:
    """Run the structural submission-format check; return a plain dict.

    Returns ``{"passed": bool, "findings": [ {...}, ... ]}`` where each
    finding carries ``check_id``, ``kind``, ``message``, ``severity`` and
    ``fix_hint``. Wrapped in a fail-closed guard: any exception becomes a
    single ``check_error`` finding with ``passed=False`` — never raised.
    """
    try:
        workdir = Path(workdir)
        config = dict(config or {})

        capsule_dir = _find_capsule_dir(workdir)

        if capsule_dir is not None:
            rows = _read_task_rows(capsule_dir)
            benchmark = rows[0].get("benchmark") if rows else None
            expected_task_ids = [r["task_id"] for r in rows if "task_id" in r]
        else:
            benchmark = config.get("benchmark")
            expected_task_ids = None

        benchmark_known = benchmark is not None
        benchmark_arg = benchmark or ""

        override = config.get("submission_file")
        if override:
            rel_names = [override]
        else:
            # `submission/submission.json` is the canonical default (matches
            # the paper's benchmark convention → zero cohort config);
            # `submission.json` at the root is a tolerant fallback.
            rel_names = ["submission/submission.json", "submission.json"]

        search_roots = [workdir]
        if capsule_dir is not None and capsule_dir != workdir:
            search_roots.append(capsule_dir)
        candidates = [root / name for name in rel_names for root in search_roots]
        default_path = workdir / rel_names[0]
        submission_path = next(
            (c for c in candidates if c.exists()), default_path
        )

        result = validate_submission(
            benchmark_arg, submission_path, expected_task_ids=expected_task_ids
        )

        findings: list[dict] = []
        emitted_unknown_info = False
        for e in result["errors"]:
            kind = e["kind"]
            if not benchmark_known and kind == "bad_task_id":
                if not emitted_unknown_info:
                    findings.append(
                        {
                            "check_id": CHECK_ID,
                            "kind": "benchmark_unknown",
                            "message": (
                                "capsule task.jsonl not found; validated "
                                "structure only (task_id shape unchecked)"
                            ),
                            "severity": "info",
                            "fix_hint": (
                                "run the gate against the bound capsule-NNN/ "
                                "workdir so task.jsonl is present"
                            ),
                        }
                    )
                    emitted_unknown_info = True
                continue
            severity = "warning" if kind in WARN_KINDS else "error"
            findings.append(
                {
                    "check_id": CHECK_ID,
                    "kind": kind,
                    "message": e["message"],
                    "severity": severity,
                    "fix_hint": FIX_HINTS.get(kind, ""),
                }
            )

        passed = not any(f["severity"] == "error" for f in findings)
        return {"passed": passed, "findings": findings}
    except Exception as exc:  # noqa: BLE001 — FAIL-CLOSED by contract.
        return {
            "passed": False,
            "findings": [
                {
                    "check_id": CHECK_ID,
                    "kind": "check_error",
                    "message": str(exc),
                    "severity": "error",
                    "fix_hint": (
                        "internal gate error; inspect the capsule workdir + "
                        "submission file"
                    ),
                }
            ],
        }


__all__ = ["CHECK_ID", "FIX_HINTS", "build_gate_result"]

# EOF
