#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_score.py

"""Correctness scoring against the operator oracle (HOST-SIDE ONLY).

:func:`score_submission` joins an agent submission to the operator
``eval/answers.jsonl`` by FULL ``task_id`` and returns one record per
oracle task::

    [{"task_id", "submitted", "expected", "verdict", "malformed_kind"?, "hint"?}]

``verdict`` is one of five values:

- ``correct`` / ``wrong`` — an eval-family comparison of a schema-valid,
  non-abstain, gradeable answer;
- ``abstain`` — an honest agent no-decision (``answer`` null or the
  canonical "cannot determine…" text, paired with a ``reason`` starting
  "agent abstained"); NOT penalised;
- ``malformed`` — a failure to produce a gradeable answer, carrying
  ``malformed_kind`` ∈ {no_submission, schema_invalid, unparseable,
  empty} when derivable from the validate stage;
- ``needs_rubric`` — a rubric-family task (BioMysteryBench) the
  mechanical scorer cannot grade; carries a ``hint``. NEVER fabricated
  into correct/wrong and NEVER folded into abstain.

Eval families dispatch on the oracle expected-answer type (mirroring
paper-scitex-clew ``_score_a.py`` at ``2bc727526``):

- **numeric** — a task_id with N reference values scores against a
  Student-t 95 % prediction interval; ``n == 1`` or a degenerate
  (zero-width) interval falls back to :func:`._sigfig.is_close_sigfig`.
- **string** — case- and whitespace-insensitive equality.
- **set-equality** — order-insensitive ``set(map(str, ...))`` equality
  with an ordered ``==`` fallback for unhashable elements.
- **rubric** — routed to ``needs_rubric``.

This is HOST-SIDE only (it reads the never-mounted oracle). The
oracle-free structural check lives in :mod:`._validate`, which this
module depends on (``_score`` → ``_validate``, never the reverse).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ._sigfig import is_close_sigfig
from ._validate import validate_submission

# Canonical abstention protocol (ported from clew ``_is_agent_abstention``).
_ABSTAIN_ANSWER_TEXT = "cannot determine from available evidence"
_ABSTAIN_REASON_PREFIX = "agent abstained"

_RUBRIC_HINT = (
    "rubric family: the mechanical scorer cannot grade this task; route "
    "the submitted answer to a rubric / LLM-judge grader against the "
    "expected rubric."
)

# Student's t critical value at p=0.975 (two-tailed 95%) for df=1..30;
# asymptotic 1.96 beyond. Mirrors scipy.stats.t.ppf(0.975, df).
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T_CRIT_975.get(df, 1.960)


def _native_from_task_id(task_id: Any) -> str | None:
    """``corebench/capsule-7038571__hard__q0`` -> ``capsule-7038571``.

    Drops an optional ``<benchmark>/`` prefix, then takes everything
    before the first ``__``. Ported from clew as a defensive fallback
    join key; the primary join is on the FULL task_id.
    """
    if not isinstance(task_id, str):
        return None
    tid = task_id.strip()
    if "/" in tid:
        tid = tid.split("/", 1)[1]
    head = tid.split("__", 1)[0].strip()
    return head or None


def _to_float(x: Any) -> float | None:
    """Coerce ``x`` to float, or ``None`` if it is not numeric."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Eval families — the CANONICAL comparators. score_submission and the
# bundled evaluate.py agree on numeric/string; set-equality is added here
# as a first-class 4th family (evaluate.py's set mode is a follow-up,
# blocked by the 512-line edit guard on _standardize.py — see PR notes).
# ---------------------------------------------------------------------------


def score_numeric(values: list[float], reported: float) -> bool:
    """95% prediction interval; ``n == 1`` / degenerate PI → sig-fig tol.

    ``lo, hi = mean ± t_{0.975, n-1} * sd * sqrt(1 + 1/n)``. A single
    reference value or a zero-width interval (all reruns identical)
    falls back to :func:`._sigfig.is_close_sigfig`.
    """
    n = len(values)
    if n == 0:
        return False
    if n == 1:
        return is_close_sigfig(reported, values[0])
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    t = _t_crit_95(n - 1)
    half = t * sd * math.sqrt(1 + 1 / n)
    if half == 0:
        return is_close_sigfig(reported, mean)
    return mean - half <= reported <= mean + half


def score_string(expected: Any, reported: Any) -> bool:
    """Case- and whitespace-insensitive string equality."""
    return str(reported).strip().lower() == str(expected).strip().lower()


def score_set(expected: Any, reported: Any) -> bool:
    """Order-insensitive set equality; ordered ``==`` fallback if unhashable.

    A non-list ``reported`` is never a match. The primary comparison
    string-coerces both sides so JSON scalars compare cleanly; if that
    raises (unhashable / un-stringifiable elements) we fall back to an
    ordered element-wise equality.
    """
    if not isinstance(reported, list):
        return False
    try:
        return set(map(str, reported)) == set(map(str, expected))
    except TypeError:
        return list(reported) == list(expected)


# ---------------------------------------------------------------------------
# Oracle loader + submission staging.
# ---------------------------------------------------------------------------


def _expected_value(payload: Any) -> Any:
    """Pull the comparable value out of an answers.jsonl ``answer`` payload.

    Handles the three per-benchmark shapes: corebench ``{"value": v}``,
    bixbench ``{"answer": a, "ideal": i}`` (prefers ``answer``), and any
    other dict / scalar (returned as-is). Rubric payloads
    (``{"rubric": ...}``) fall through to the raw dict and are handled
    by :func:`_is_rubric`.
    """
    if isinstance(payload, dict):
        for key in ("value", "answer", "ideal"):
            if key in payload and payload[key] is not None:
                return payload[key]
        return payload
    return payload


def _is_rubric(payload: Any, benchmark: str) -> bool:
    """True iff the oracle payload is a rubric (mechanically ungradeable)."""
    if isinstance(payload, dict) and "rubric" in payload:
        if "value" not in payload and "answer" not in payload:
            return True
    return benchmark == "biomysterybench"


def _load_oracle(answers: Path | str) -> dict[str, list[Any]]:
    """Load ``answers.jsonl`` into ``{full_task_id: [answer_payload, ...]}``.

    ``answers`` may be the ``answers.jsonl`` file itself or a directory
    holding it (the eval dir). Records are grouped by FULL ``task_id``
    preserving first-seen order, so a task_id repeated across N oracle
    lines yields N reference values for the numeric prediction interval.
    """
    p = Path(answers)
    if p.is_dir():
        p = p / "answers.jsonl"
    grouped: dict[str, list[Any]] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            tid = obj.get("task_id")
            if not isinstance(tid, str):
                continue
            grouped.setdefault(tid, []).append(obj.get("answer"))
    return grouped


def _load_submission(
    benchmark: str, submission: Any
) -> tuple[list | None, str | None]:
    """Return ``(rows, global_malformed_kind)`` for the submission.

    The ``global_malformed_kind``, when not None, applies to EVERY
    oracle task (the submission as a whole is ungradeable):
    ``no_submission`` (missing file / None), ``unparseable`` (not JSON),
    ``empty`` (parsed empty array), ``schema_invalid`` (fails
    :func:`._validate.validate_submission`). A usable submission returns
    ``(rows, None)``.
    """
    if submission is None:
        return None, "no_submission"
    if isinstance(submission, (str, Path)):
        p = Path(submission)
        if not p.exists():
            return None, "no_submission"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, "unparseable"
    else:
        data = submission

    if isinstance(data, list) and len(data) == 0:
        return data, "empty"
    result = validate_submission(benchmark, data)
    if not result["ok"]:
        return (data if isinstance(data, list) else None), "schema_invalid"
    return data, None


def _is_abstention(submitted: Any, reason: Any) -> bool:
    """Port of clew ``_is_agent_abstention`` (reason prefix is required)."""
    if not (
        isinstance(reason, str)
        and reason.strip().lower().startswith(_ABSTAIN_REASON_PREFIX)
    ):
        return False
    if submitted is None:
        return True
    if (
        isinstance(submitted, str)
        and submitted.strip().lower() == _ABSTAIN_ANSWER_TEXT
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def score_submission(
    benchmark: str,
    submission: Any,
    *,
    answers: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> list[dict]:
    """Score ``submission`` against the ``benchmark`` oracle answers.

    ``submission`` may be a path (``str`` / ``Path``) to the submission
    JSON, an already-parsed list, or ``None`` (no submission). ``answers``
    is the ``eval/answers.jsonl`` file or its directory; when omitted it
    is resolved from ``dataset_root`` via
    :func:`._base.resolve_paths`. Returns one record per oracle task_id,
    ordered by task_id.
    """
    if answers is None:
        from ._base import resolve_paths

        answers = resolve_paths(benchmark, dataset_root=dataset_root).eval_dir

    oracle = _load_oracle(answers)
    rows, global_kind = _load_submission(benchmark, submission)

    by_id: dict[str, dict] = {}
    if global_kind is None and isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("task_id"), str):
                by_id[row["task_id"]] = row

    results: list[dict] = []
    for task_id in sorted(oracle):
        payloads = oracle[task_id]
        first = payloads[0] if payloads else None
        rec: dict = {
            "task_id": task_id,
            "submitted": None,
            "expected": _expected_value(first),
        }

        # Whole-submission failure applies to every task uniformly.
        if global_kind is not None:
            rec["verdict"] = "malformed"
            rec["malformed_kind"] = global_kind
            results.append(rec)
            continue

        row = by_id.get(task_id)
        if row is None:
            rec["verdict"] = "malformed"
            rec["malformed_kind"] = "no_submission"
            results.append(rec)
            continue

        submitted = row.get("answer")
        reason = row.get("reason")
        rec["submitted"] = submitted

        # Honest abstention precedes the malformed / rubric routing.
        if _is_abstention(submitted, reason):
            rec["verdict"] = "abstain"
            results.append(rec)
            continue

        # Rubric family: mechanically ungradeable -> needs_rubric (+hint).
        if _is_rubric(first, benchmark):
            rec["verdict"] = "needs_rubric"
            rec["expected"] = (
                first.get("rubric") if isinstance(first, dict) else rec["expected"]
            )
            rec["hint"] = _RUBRIC_HINT
            results.append(rec)
            continue

        # Bare null with no abstain reason is a malformed (empty) answer.
        if submitted is None:
            rec["verdict"] = "malformed"
            rec["malformed_kind"] = "empty"
            results.append(rec)
            continue

        # Eval-family dispatch on the oracle expected type.
        values = [_expected_value(p) for p in payloads]
        sample = values[0]
        rec["expected"] = sample
        if isinstance(sample, list):
            ok = score_set(sample, submitted)
            rec["verdict"] = "correct" if ok else "wrong"
        elif isinstance(sample, bool):
            ok = score_string(sample, submitted)
            rec["verdict"] = "correct" if ok else "wrong"
        elif isinstance(sample, (int, float)):
            nums = [f for f in (_to_float(v) for v in values) if f is not None]
            rep = _to_float(submitted)
            if rep is None or not nums:
                # Non-numeric answer to a numeric task: not gradeable.
                rec["verdict"] = "malformed"
            else:
                ok = score_numeric(nums, rep)
                rec["verdict"] = "correct" if ok else "wrong"
        elif isinstance(sample, str):
            ok = score_string(sample, submitted)
            rec["verdict"] = "correct" if ok else "wrong"
        else:
            rec["verdict"] = "malformed"
        results.append(rec)

    return results


__all__ = [
    "score_submission",
    "score_numeric",
    "score_string",
    "score_set",
]

# EOF
