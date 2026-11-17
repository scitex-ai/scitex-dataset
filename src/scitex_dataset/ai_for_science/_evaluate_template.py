#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_evaluate_template.py

"""Shared stdlib-only ``evaluate.py`` scorer template (split from ``_standardize``)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# evaluate.py — a single shared, stdlib-only, answer-free CLI scorer.
# ``{default_mode}`` and ``{reference_field}`` are substituted per benchmark
# at write time.
# ---------------------------------------------------------------------------

EVALUATE_PY = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uniform answer-free scorer for an ai-for-science benchmark.

Usage::

    python evaluate.py --submission S.json --answers answers.jsonl \\
        [--mode numeric|string|rubric]

Loads the agent submission (the uniform ``[{{task_id, answer}}, ...]``
array) and the operator ``answers.jsonl``, joins them by ``task_id``,
scores each task by the selected ``--mode``, and prints a JSON summary::

    {{"n", "n_scored", "n_correct", "score", "per_task": [...]}}

The scoring logic is answer-free: the answers come from the ``--answers``
file argument, never from this source. Stdlib-only so it runs anywhere.

Modes:
- ``numeric`` — parse both to float, correct within relative tol 1e-3.
- ``string``  — case- and whitespace-normalized equality after legacy
  non-null string coercion; missing/null submissions are graded failures.
  A ``(lo,hi)`` range reference is graded as inclusive interval membership
  and a numeric-text reference with a numeric answer by the 1e-3 tolerance.
- ``rubric``  — not auto-scorable: each task is marked
  ``"requires_rubric_grading"`` and excluded from ``n_scored`` / ``score``.

``REFERENCE_FIELD`` (when set) names the ONLY payload key graded; a record
missing it aborts with an error rather than being graded against another
key, and a null / blank reference is reported as ``invalid_reference``.

This is distinct from the package API's pooled-reference PI / sig-fig
policy. Repeated references are retained and count as ONE question, but
scalar comparison refuses them as ``needs_reference_policy`` until an
explicit aggregation policy is selected. No first/last/mean is assumed.
Malformed/missing solver answers remain scored failures when the reference
is gradeable. An entirely ungradeable assignment has ``score: null``.
"""

import argparse
import json
import math
import re
import sys

DEFAULT_MODE = "{default_mode}"
REFERENCE_FIELD = {reference_field!r}
_REL_TOL = 1e-3


def _load_submission(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("submission must be an array")
    by_id = {{}}
    for row in data:
        if (not isinstance(row, dict) or not isinstance(row.get("task_id"), str)
                or not row["task_id"].strip() or "answer" not in row):
            raise ValueError("invalid submission record")
        if row["task_id"] in by_id:
            raise ValueError("duplicate submission task_id")
        by_id[row["task_id"]] = row.get("answer")
    return by_id


def _load_answers(path):
    by_id = {{}}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (not isinstance(rec, dict) or not isinstance(rec.get("task_id"), str)
                    or not rec["task_id"].strip()):
                raise ValueError("invalid reference record")
            if "answer" not in rec:
                raise ValueError("reference record missing answer field")
            by_id.setdefault(rec["task_id"], []).append(rec)
    return by_id


def _coerce_answer_value(answer_rec):
    """Pull the comparable scalar out of an answers.jsonl ``answer`` field.

    The operator answer payload is a small dict (e.g. ``{{"value": ...}}``,
    ``{{"answer": ..., "ideal": ...}}``, ``{{"rubric": ...}}``). We prefer
    the first non-null recognized key. An all-null recognized wrapper yields
    null; a dict without any recognized keys remains the raw value.
    """
    ans = answer_rec.get("answer")
    if isinstance(ans, dict) and REFERENCE_FIELD is not None:
        if REFERENCE_FIELD not in ans:
            raise ValueError("reference record %r: field %r is missing"
                             % (answer_rec.get("task_id"), REFERENCE_FIELD))
        value = ans[REFERENCE_FIELD]
        if isinstance(value, str) and not value.strip():
            return None  # blank reference: reported as invalid_reference
        return value
    if isinstance(ans, dict):
        for key in ("value", "answer", "ideal", "rubric"):
            if key in ans and ans[key] is not None:
                return ans[key]
        if any(key in ans for key in ("value", "answer", "ideal", "rubric")):
            return None
        return ans
    return ans


def _finite_float(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _score_numeric(submitted, expected):
    s = _finite_float(submitted)
    e = _finite_float(expected)
    if s is None or e is None:
        return False
    if e == 0:
        return abs(s) <= _REL_TOL
    return abs(s - e) <= _REL_TOL * abs(e)


_NUM = r"[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?"
_INTERVAL_RE = re.compile(r"^\\s*[\\(\\[]\\s*(" + _NUM + r")\\s*,\\s*(" + _NUM + r")\\s*[\\)\\]]\\s*$")


def _interval(value):
    m = _INTERVAL_RE.match(value) if isinstance(value, str) else None
    if m is None:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (min(lo, hi), max(lo, hi))


def _norm_string(value):
    return " ".join(str(value).strip().lower().split())


def _score_string(submitted, expected):
    interval = _interval(expected)
    if interval is not None:
        s = _finite_float(submitted)
        return s is not None and interval[0] <= s <= interval[1]
    if _finite_float(expected) is not None and _finite_float(submitted) is not None:
        # Numeric TEXT reference (e.g. BixBench ``ideal``): scalar tolerance.
        return _score_numeric(submitted, expected)
    return _norm_string(submitted) == _norm_string(expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument(
        "--mode",
        choices=("numeric", "string", "rubric"),
        default=DEFAULT_MODE,
    )
    args = parser.parse_args(argv)

    submitted = _load_submission(args.submission)
    answers = _load_answers(args.answers)

    per_task = []
    n_scored = 0
    n_correct = 0
    for task_id, reference_records in sorted(answers.items()):
        sub = submitted.get(task_id)
        n_references = len(reference_records)
        if args.mode == "rubric":
            per_task.append(
                {{
                    "task_id": task_id,
                    "status": "requires_rubric_grading",
                    "submitted": sub,
                    "n_references": n_references,
                }}
            )
            continue
        if n_references != 1:
            per_task.append({{"task_id": task_id,
                             "status": "needs_reference_policy",
                             "n_references": n_references, "submitted": sub}})
            continue
        answer_rec = reference_records[0]
        expected = _coerce_answer_value(answer_rec)
        if args.mode == "numeric":
            if _finite_float(expected) is None:
                per_task.append({{"task_id": task_id, "status": "invalid_reference",
                                 "n_references": n_references}})
                continue
            if _finite_float(sub) is None:
                n_scored += 1
                status = "malformed" if task_id in submitted else "no_submission"
                per_task.append({{"task_id": task_id, "status": status,
                                 "correct": False, "n_references": n_references,
                                 "submitted": sub}})
                continue
            correct = _score_numeric(sub, expected)
        else:
            if expected is None:
                per_task.append({{"task_id": task_id, "status": "invalid_reference",
                                 "n_references": n_references}})
                continue
            if task_id not in submitted or sub is None:
                n_scored += 1
                status = "malformed" if task_id in submitted else "no_submission"
                per_task.append({{"task_id": task_id, "status": status,
                                 "correct": False, "n_references": n_references,
                                 "submitted": sub}})
                continue
            correct = _score_string(sub, expected)
        n_scored += 1
        if correct:
            n_correct += 1
        per_task.append(
            {{"task_id": task_id, "correct": bool(correct), "submitted": sub,
              "n_references": n_references}}
        )

    summary = {{
        "n": len(answers),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_ungradeable": len(answers) - n_scored,
        "policy": "scalar-1e-3/string-whitespace-v1; multisample-unsupported",
        "score": (n_correct / n_scored) if n_scored else None,
        "per_task": per_task,
    }}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_evaluate_py(default_mode: str, reference_field: str | None = None) -> str:
    """Return :data:`EVALUATE_PY` with the mode and reference key substituted.

    ``reference_field`` is the adapter's ``REFERENCE_FIELD`` (the only payload
    key graded); ``None`` keeps the legacy first-non-null key search.
    """
    return EVALUATE_PY.format(default_mode=default_mode, reference_field=reference_field)


__all__ = ["EVALUATE_PY", "render_evaluate_py"]

# EOF
