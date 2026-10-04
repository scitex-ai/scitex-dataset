#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_gate.py

"""Pure, scitex_dev-agnostic, oracle-free submission-format gate.

Only the selected public task contract provides IDs, declared benchmark and
optional answer types. Finding dataclass fields remain unchanged: public
field/task/path guidance is rendered in message/fix_hint. A workdir containing
its own task.jsonl is explicitly bound; otherwise exactly one capsule child
may be discovered. Multiple candidates refuse rather than choosing the first.

No-task fallback is explicitly shape-only, even with a configured benchmark.
Malformed public metadata and unexpected errors fail closed; exception/input
payloads are never copied into feedback. This does not score answers or set
scientific precision, abstention, missingness or retry policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._validate import WARN_KINDS, read_public_task_contract, validate_submission

CHECK_ID = "dataset-submission-format"
FIX_HINTS: dict[str, str] = {
    "no_file": "Write your JSON array to `submission/submission.json` in the capsule workdir (see submission.example.json).",
    "unreadable": "Make the submission file readable UTF-8 JSON, then retry the format check.",
    "unparseable": 'Write a JSON array of {"task_id", "answer"} objects.',
    "wrong_type": "Use the declared public field types and supported JSON values; the submission must be an array of objects. No type coercion is performed.",
    "wrong_answer_type": "Use the answer_type declared in public task.jsonl, or null with a nonempty reason to abstain. No type coercion is performed.",
    "missing_field": "Each item needs both 'task_id' and 'answer'.",
    "missing_reason": "A null answer requires a nonempty one-line reason naming the proximal cause, or provide an answer.",
    "bad_task_id": "Use the exact task_id(s) from the selected task.jsonl/submission.example.json.",
    "unknown_field": "Extra JSON fields are retained as warnings; you may remove unused fields other than task_id, answer and reason.",
    "wrong_count": "Submit exactly one item per assigned task_id in this capsule.",
    "duplicate_task_id": "Keep exactly one submission item per assigned task_id; retain every other assigned task or explicitly abstain.",
    "unknown_task_id": "Use IDs from the selected public task.jsonl, not a different capsule or historical assignment.",
    "missing_task_id": "Add the named assigned task once, with its answer or null plus a nonempty reason.",
    "duplicate_assignment": "Repair duplicate IDs in the public assignment; do not silently deduplicate task or reference units.",
    "invalid_assignment": "Use valid, consistent public task.jsonl metadata for the selected capsule. Do not substitute evaluator references.",
    "missing_assignment": "Run the gate against the selected capsule containing public task.jsonl.",
    "ambiguous_capsule": "Run the gate inside one selected capsule-NNN/ directory; do not pass a parent holding multiple capsules.",
    "shape_only": "Run the gate against the selected capsule with public task.jsonl to check membership and declared answer types.",
}


class _AmbiguousCapsuleError(ValueError):
    pass


def _find_capsule_dir(workdir: Path) -> Path | None:
    """Honor a directly bound task file or discover one unambiguous child."""
    if (workdir / "task.jsonl").is_file():
        return workdir
    candidates = [path for path in sorted(workdir.glob("capsule-*"))
                  if path.is_dir() and (path / "task.jsonl").is_file()]
    if len(candidates) > 1:
        raise _AmbiguousCapsuleError("multiple public capsule task files")
    return candidates[0] if candidates else None


def _finding(kind: str, message: str, severity: str = "error", *, fix_hint: str | None = None) -> dict:
    return {"check_id": CHECK_ID, "kind": kind, "message": message,
            "severity": severity, "fix_hint": FIX_HINTS.get(kind, "Repair the public submission format and retry.") if fix_hint is None else fix_hint}


def _format_findings(errors: list[dict], *, benchmark_known: bool = True) -> list[dict]:
    findings = []
    for error in errors:
        kind = error["kind"]
        if not benchmark_known and kind == "bad_task_id":
            continue
        task = f" [task {error['task_id']}]" if "task_id" in error else ""
        message = f"{error['path']}{task}: {error['message']}"
        hint = FIX_HINTS.get(kind)
        if kind == "wrong_answer_type":
            # The validator message contains only the public declared type,
            # never a reference or submitted answer value.
            hint = error["message"]
        findings.append(_finding(kind, message, "warning" if kind in WARN_KINDS else "error", fix_hint=hint))
    return findings


def build_gate_result(workdir: Any, config: Mapping | None) -> dict:
    """Return GateResult-compatible fields without importing its engine."""
    try:
        workdir = Path(workdir)
        config = dict(config or {})
        capsule_dir = _find_capsule_dir(workdir)
        if capsule_dir is not None:
            contract = read_public_task_contract(capsule_dir)
            if not contract["ok"]:
                return {"passed": False, "findings": _format_findings(contract["errors"])}
            benchmark = contract["benchmark"] or config.get("benchmark")
            if not (isinstance(benchmark, str) and benchmark.strip()):
                return {"passed": False, "findings": [_finding("invalid_assignment", "$assignment.benchmark: selected public tasks need a consistent benchmark declaration or explicit benchmark config")]}
            expected_ids = contract["task_ids"]
            expected_types = contract["answer_types"]
        else:
            benchmark = config.get("benchmark")
            expected_ids = expected_types = None
        if benchmark is not None and not (isinstance(benchmark, str) and benchmark.strip()):
            return {"passed": False, "findings": [_finding("invalid_assignment", "$config.benchmark: configured benchmark must be a nonempty string")]}
        benchmark_known = isinstance(benchmark, str) and bool(benchmark.strip())
        benchmark_arg = benchmark if benchmark_known else ""
        override = config.get("submission_file")
        rel_names = [override] if override else ["submission/submission.json", "submission.json"]
        roots = [workdir]
        if capsule_dir is not None and capsule_dir != workdir:
            roots.append(capsule_dir)
        candidates = [root / name for name in rel_names for root in roots]
        submission_path = next((path for path in candidates if path.exists()), workdir / rel_names[0])
        result = validate_submission(benchmark_arg, submission_path,
                                     expected_task_ids=expected_ids,
                                     expected_answer_types=expected_types)
        findings = _format_findings(result["errors"], benchmark_known=benchmark_known)
        if capsule_dir is None:
            findings.append(_finding("shape_only", "No public task.jsonl: checked JSON structure and any configured benchmark syntax only; selected membership and public answer types were not checked.", "info"))
            if not benchmark_known:
                findings.append(_finding("benchmark_unknown", "No public benchmark declaration or config: task_id benchmark syntax was not checked.", "info", fix_hint=FIX_HINTS["shape_only"]))
        return {"passed": not any(f["severity"] == "error" for f in findings), "findings": findings}
    except _AmbiguousCapsuleError:
        return {"passed": False, "findings": [_finding("ambiguous_capsule", "Multiple capsule task files found; selected assignment is ambiguous.")]}
    except Exception:  # noqa: BLE001 — fail closed, without exception payloads.
        return {"passed": False, "findings": [_finding("check_error", "Submission-format gate could not complete; no answer or exception payload was exposed.", fix_hint="Inspect the public capsule metadata and submission file; retry only after resolving the format or access error.")]}


__all__ = ["CHECK_ID", "FIX_HINTS", "build_gate_result"]

# EOF
