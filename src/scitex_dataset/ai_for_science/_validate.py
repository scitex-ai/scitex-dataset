#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_validate.py

"""Strict, oracle-free submission validation and public assignment loading.

Pydantic checks supported JSON values without scientific type coercion.
Answers and extra fields are never rewritten. Extra fields remain warnings;
a null answer still requires a nonempty string reason. A non-null answer
keeps the legacy optional reason behavior.

Exact selected task membership is independent of scientific scoring. Public
``task.jsonl`` may optionally declare ``answer_type`` as ``json`` (default),
``number``, ``integer``, ``string``, ``boolean``, ``list`` or ``object``.
Absent declarations accept any supported JSON answer; they do not imply that
an answer is scientifically numeric. Types are never inferred from references.

This module does not load evaluator answers or import a scorer. Feedback
contains public field/type requirements and task IDs, never answer values or
closeness hints. CORE accepts the new question-hash ID shape and the legacy ID
shape for structural API compatibility; exact assignment membership never
aliases the two. Format validation does not change abstention, tolerance,
scoring, experiment retry, or assigned-denominator policies.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

# Writer/schema constant only; NEVER import a scorer or reference reader here.
from ._standardize import UNIFORM_SUBMISSION_SCHEMA

_KNOWN_ITEM_KEYS = {"task_id", "answer", "reason"}
WARN_KINDS = {"unknown_field", "shape_only"}
VALIDATION_CONTRACT = "submission-validation-v2"
AnswerType = Literal["json", "number", "integer", "string", "boolean", "list", "object"]
_STRICT_JSON = ConfigDict(strict=True, allow_inf_nan=False)


class SubmissionRow(BaseModel):
    """Required submission fields; extra JSON fields are retained as warnings."""

    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue]
    task_id: StrictStr
    answer: JsonValue
    # For answered rows, preserve the existing optional JSON reason payload.
    # The null-answer rule is checked separately for an exact field diagnostic.
    reason: JsonValue = None


class _PublicTask(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    task_id: StrictStr
    answer_type: AnswerType = "json"


class _PublicIndexRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    task_ids: list[StrictStr]


_SUBMISSION = TypeAdapter(list[SubmissionRow], config=_STRICT_JSON)
_PUBLIC_ANSWER_TYPES = TypeAdapter(dict[StrictStr, AnswerType], config=_STRICT_JSON)
_ANSWER_TYPES = {
    "json": TypeAdapter(JsonValue, config=_STRICT_JSON),
    "number": TypeAdapter(StrictInt | StrictFloat, config=_STRICT_JSON),
    "integer": TypeAdapter(StrictInt, config=_STRICT_JSON),
    "string": TypeAdapter(StrictStr, config=_STRICT_JSON),
    "boolean": TypeAdapter(StrictBool, config=_STRICT_JSON),
    "list": TypeAdapter(list[JsonValue], config=_STRICT_JSON),
    "object": TypeAdapter(dict[str, JsonValue], config=_STRICT_JSON),
}


def _err(path: str, kind: str, message: str, *, task_id: str | None = None) -> dict:
    error = {"path": path, "kind": kind, "message": message}
    if task_id is not None:
        error["task_id"] = task_id
    if "." in path:
        error["field"] = path.rsplit(".", 1)[1]
    return error


def _coerce(submission: Any) -> tuple[Any, dict | None]:
    """Read a submission path or retain the already-parsed object unchanged."""
    if isinstance(submission, (str, Path)):
        p = Path(submission)
        if not p.exists():
            return None, _err("$", "no_file", f"submission file not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8")), None
        except (json.JSONDecodeError, UnicodeError):
            return None, _err("$", "unparseable", "submission is not valid UTF-8 JSON")
        except OSError:
            return None, _err("$", "unreadable", "submission file cannot be read")
    return submission, None


def _valid_task_id(benchmark: str, task_id: str) -> bool:
    """Check shape only; exact public assignment membership is separate."""
    prefix = f"{benchmark}/"
    if not task_id.startswith(prefix):
        return False
    rest = task_id[len(prefix):]
    if not rest:
        return False
    if benchmark == "corebench":
        native, separator, digest = rest.rpartition("__question_")
        if separator:
            return bool(native) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        parts = rest.split("__")
        if len(parts) != 3:
            return False
        native, difficulty, question = parts
        return bool(native and difficulty) and re.fullmatch(r"q[0-9]+", question) is not None
    return True


class PublicTaskContractError(ValueError):
    """A present public assignment is malformed; do not disable membership."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("public assignment metadata is invalid")


def read_public_task_contract(source: Path | str) -> dict:
    """Read an explicit public task.jsonl or index.jsonl, never evaluator rows.

    For a directory, prefer its selected capsule ``task.jsonl``; otherwise
    read its whole-catalog ``index.jsonl``. Blank separators are harmless.
    Malformed or duplicate assigned records are attributable hard errors.
    The index is a catalog scope, not an inferred selected capsule.
    """
    path = Path(source)
    if path.is_dir() or (not path.exists() and not path.suffix):
        selected = path / "task.jsonl"
        path = selected if selected.is_file() else path / "index.jsonl"
    result = {"ok": False, "errors": [], "task_ids": [], "answer_types": {},
              "source": str(path), "scope": "selected" if path.name == "task.jsonl" else "catalog", "benchmark": None}
    if path.name not in {"task.jsonl", "index.jsonl"}:
        result["errors"].append(_err("$assignment", "invalid_assignment", "use public task.jsonl or index.jsonl, not evaluator files"))
        return result
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        result["errors"].append(_err("$assignment", "missing_assignment", "public assignment file is missing; supply the selected task.jsonl"))
        return result
    except (OSError, UnicodeError):
        result["errors"].append(_err("$assignment", "invalid_assignment", "public assignment file cannot be read as UTF-8"))
        return result
    first_locations: dict[str, str] = {}
    benchmarks: list[str | None] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        location = f"$assignment[{line_number}]"
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            result["errors"].append(_err(location, "invalid_assignment", "assignment row must be valid JSON"))
            continue
        try:
            if path.name == "task.jsonl":
                task = _PublicTask.model_validate(row, strict=True)
                ids = [task.task_id]
                answer_types = {task.task_id: task.answer_type}
                benchmark = row.get("benchmark")
                if "benchmark" in row and not (isinstance(benchmark, str) and benchmark.strip()):
                    result["errors"].append(_err(f"{location}.benchmark", "invalid_assignment", "public benchmark must be a nonempty string when declared"))
                benchmarks.append(benchmark)
            else:
                task = _PublicIndexRow.model_validate(row, strict=True)
                ids = task.task_ids
                answer_types = {}
        except ValidationError as exc:
            # Never stringify ValidationError: it includes input values.
            for error in exc.errors(include_input=False, include_context=False, include_url=False):
                field = error["loc"][0] if error["loc"] else None
                field_path = f"{location}.{field}" if isinstance(field, str) else location
                result["errors"].append(_err(field_path, "invalid_assignment", "assignment row requires a string task_id and declared answer_type, or an index array of string task_ids"))
            continue
        for task_id in ids:
            if not task_id:
                result["errors"].append(_err(location, "invalid_assignment", "assigned task_id must be nonempty"))
            elif task_id in first_locations:
                result["errors"].append(_err(location, "duplicate_assignment", f"assigned task_id occurs more than once; first at {first_locations[task_id]}", task_id=task_id))
            else:
                first_locations[task_id] = location
            result["task_ids"].append(task_id)
        result["answer_types"].update(answer_types)
    if benchmarks:
        declared = {value for value in benchmarks if isinstance(value, str) and value.strip()}
        if len(declared) > 1 or (declared and any(value is None for value in benchmarks)):
            result["errors"].append(_err("$assignment.benchmark", "invalid_assignment", "declare the same benchmark on every selected task, or omit it on every task"))
        elif len(declared) == 1:
            result["benchmark"] = next(iter(declared))
    result["ok"] = not result["errors"]
    return result


def expected_task_ids_from_for_solver(for_solver_dir: Path | str) -> list[str] | None:
    """Compatibility helper; missing metadata returns None, malformed raises.

    Prefer selected ``task.jsonl`` when present. Empty valid assignments return
    an empty list rather than silently skipping their membership check.
    """
    root = Path(for_solver_dir)
    if not root.exists() or (root.is_dir() and not any((root / name).is_file() for name in ("task.jsonl", "index.jsonl"))):
        return None
    contract = read_public_task_contract(root)
    if not contract["ok"]:
        raise PublicTaskContractError(contract["errors"])
    return contract["task_ids"]


def _shape_errors(data: Any) -> list[dict]:
    try:
        _SUBMISSION.validate_python(data, strict=True)
        return []
    except ValidationError as exc:
        findings = []
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            loc = error["loc"]
            index = loc[0] if loc and isinstance(loc[0], int) else None
            field = loc[1] if len(loc) > 1 and isinstance(loc[1], str) else None
            path = f"$[{index}]" if index is not None else "$"
            if field is not None:
                path += f".{field}"
            task_id = None
            if isinstance(data, list) and index is not None and isinstance(data[index], dict):
                if isinstance(data[index].get("task_id"), str):
                    task_id = data[index]["task_id"]
            if error["type"] == "missing":
                kind, message = "missing_field", f"item is missing '{field}'"
            elif field == "task_id":
                kind, message = "wrong_type", "'task_id' must be a string; no coercion is performed"
            elif field is not None:
                kind, message = "wrong_type", f"'{field}' must contain supported JSON values (finite numbers, strings, booleans, null, arrays or objects); no coercion is performed"
            else:
                kind, message = "wrong_type", "submission must be a JSON array of objects" if index is None else "submission item must be an object"
            findings.append(_err(path, kind, message, task_id=task_id))
        return findings


def validate_submission(
    benchmark: str,
    submission: Any,
    *,
    expected_task_ids: list[str] | None = None,
    expected_answer_types: dict[str, AnswerType] | None = None,
    require_complete: bool = True,
) -> dict:
    """Return deterministic public repair feedback without grading answers.

    Optional expected IDs specify the exact selected assignment. Optional
    answer types must come from that public assignment, never its references.
    Null with a nonempty reason remains allowed for every declared answer type.
    The input is validated, not rewritten; unknown JSON fields are warnings.
    ``require_complete=False`` permits host scoring of partial submissions:
    uniqueness/membership still apply, while missing tasks remain scorer-owned.
    """
    data, load_error = _coerce(submission)
    if load_error is not None:
        return {"ok": False, "errors": [load_error]}
    errors = _shape_errors(data)
    if expected_answer_types is not None:
        try:
            _PUBLIC_ANSWER_TYPES.validate_python(expected_answer_types, strict=True)
        except ValidationError:
            errors.append(_err("$assignment.answer_type", "invalid_assignment", "public answer_type declarations require string task IDs and json, number, integer, string, boolean, list or object"))
            expected_answer_types = None
    if not isinstance(data, list):
        return {"ok": False, "errors": errors}
    seen: dict[str, int] = {}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        path = f"$[{index}]"
        task_id = item.get("task_id")
        if isinstance(task_id, str):
            if not _valid_task_id(benchmark, task_id):
                errors.append(_err(f"{path}.task_id", "bad_task_id", f"'task_id' does not match the {benchmark} public ID shape", task_id=task_id))
            if task_id in seen:
                errors.append(_err(f"{path}.task_id", "duplicate_task_id", f"submit each assigned task once; first occurrence is $[{seen[task_id]}].task_id", task_id=task_id))
            else:
                seen[task_id] = index
        if "answer" in item and item["answer"] is None:
            reason = item.get("reason")
            if not (isinstance(reason, str) and reason.strip()):
                label = task_id if isinstance(task_id, str) else path
                errors.append(_err(f"{path}.reason", "missing_reason", f"question {label}: answer is null but has no reason — honest abstention requires a one-line reason", task_id=task_id if isinstance(task_id, str) else None))
        elif "answer" in item and isinstance(task_id, str) and expected_answer_types and task_id in expected_answer_types:
            declared = expected_answer_types[task_id]
            adapter = _ANSWER_TYPES.get(declared) if isinstance(declared, str) else None
            if adapter is None:
                errors.append(_err("$assignment.answer_type", "invalid_assignment", "public answer_type must be json, number, integer, string, boolean, list or object", task_id=task_id))
            else:
                try:
                    adapter.validate_python(item["answer"], strict=True)
                except ValidationError:
                    errors.append(_err(f"{path}.answer", "wrong_answer_type", f"public assignment requires answer_type '{declared}'; no coercion is performed, or use null with a nonempty reason to abstain", task_id=task_id))
        for key in item:
            if key not in _KNOWN_ITEM_KEYS:
                errors.append(_err(f"{path}.{key}", "unknown_field", f"unknown field {key!r} (retained as a warning)", task_id=task_id if isinstance(task_id, str) else None))
    if expected_task_ids is not None:
        if not isinstance(expected_task_ids, list) or any(not isinstance(tid, str) or not _valid_task_id(benchmark, tid) for tid in expected_task_ids):
            errors.append(_err("$assignment", "invalid_assignment", "expected task IDs must be an array of valid public benchmark IDs"))
        else:
            duplicates = Counter(expected_task_ids)
            for task_id, count in duplicates.items():
                if count > 1:
                    errors.append(_err("$assignment", "duplicate_assignment", "assigned task IDs must be unique", task_id=task_id))
            expected = set(expected_task_ids)
            if require_complete and len(data) != len(expected_task_ids):
                errors.append(_err("$", "wrong_count", f"submission has {len(data)} item(s); expected {len(expected_task_ids)}"))
            for task_id, index in seen.items():
                if task_id not in expected:
                    errors.append(_err(f"$[{index}].task_id", "unknown_task_id", "task_id is not in the selected public assignment", task_id=task_id))
            for task_id in dict.fromkeys(expected_task_ids):
                if require_complete and task_id not in seen:
                    errors.append(_err("$", "missing_task_id", "submit this assigned task once, or explicitly abstain with null and a nonempty reason", task_id=task_id))
    return {"ok": not any(error["kind"] not in WARN_KINDS for error in errors), "errors": errors}


_SCHEMA = UNIFORM_SUBMISSION_SCHEMA
__all__ = ["validate_submission", "read_public_task_contract", "expected_task_ids_from_for_solver", "PublicTaskContractError", "SubmissionRow", "VALIDATION_CONTRACT", "WARN_KINDS"]

# EOF
