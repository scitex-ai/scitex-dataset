#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_standardize.py

"""Shared ``standardize`` helpers — raw oracle → {for_solver, eval}.

Every AI-for-science benchmark splits its operator-private ``raw/``
oracle (which *has* answers) into two derived views with a UNIFORM
on-disk contract, so the experiment harness treats all benchmarks
identically:

- ``for_solver/`` — the AGENT view (leak-safe). A ``tasks.jsonl`` of
  ``{task_id, benchmark, prompt, data}`` records (NO answers), a
  ``submission.schema.json`` describing the agent's expected output, a
  ``submission.example.json`` one-row example, and relative symlinks to
  the answer-free problem data under ``raw/``.
- ``eval/`` — the OPERATOR view (never mounted). An ``answers.jsonl``
  keyed by the same ``task_id`` as ``tasks.jsonl`` plus a stdlib-only
  ``evaluate.py`` CLI scorer. The scorer is answer-free logic — the
  answers are passed in as a file argument, never baked into the code.

Per-benchmark modules build the ``tasks`` + ``answers`` lists with their
own mappers, then call :func:`write_for_solver` and :func:`write_eval`.
The single shared :data:`EVALUATE_PY` template implements a uniform
scorer; each benchmark bakes its default ``--mode`` into the rendered
copy via the ``{default_mode}`` placeholder.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

# The uniform submission contract: an array of objects, each carrying the
# ``task_id`` it answers plus the agent's ``answer`` (any JSON value).
UNIFORM_SUBMISSION_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ai-for-science uniform submission",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "answer": {},
        },
        "required": ["task_id", "answer"],
        "additionalProperties": True,
    },
}

# Uniform keys every for_solver task row carries — used by callers and
# tests to assert the leak-safe schema is exactly these (no answer-
# bearing keys leak through).
TASK_KEYS = ("task_id", "benchmark", "prompt", "data")


# ---------------------------------------------------------------------------
# evaluate.py — a single shared, stdlib-only, answer-free CLI scorer.
# ``{default_mode}`` is substituted per benchmark at write time.
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
- ``string``  — case- and whitespace-normalized equality.
- ``rubric``  — not auto-scorable: each task is marked
  ``"requires_rubric_grading"`` and excluded from ``n_scored`` / ``score``.
"""

import argparse
import json
import sys

DEFAULT_MODE = "{default_mode}"
_REL_TOL = 1e-3


def _load_submission(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    by_id = {{}}
    for row in data:
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
            by_id[rec["task_id"]] = rec
    return by_id


def _coerce_answer_value(answer_rec):
    """Pull the comparable scalar out of an answers.jsonl ``answer`` field.

    The operator answer payload is a small dict (e.g. ``{{"value": ...}}``,
    ``{{"answer": ..., "ideal": ...}}``, ``{{"rubric": ...}}``). We prefer
    the most specific keys; fall back to the raw value.
    """
    ans = answer_rec.get("answer")
    if isinstance(ans, dict):
        for key in ("value", "answer", "ideal", "rubric"):
            if key in ans and ans[key] is not None:
                return ans[key]
        return ans
    return ans


def _score_numeric(submitted, expected):
    try:
        s = float(submitted)
        e = float(expected)
    except (TypeError, ValueError):
        return False
    if e == 0:
        return abs(s) <= _REL_TOL
    return abs(s - e) <= _REL_TOL * abs(e)


def _norm_string(value):
    return " ".join(str(value).strip().lower().split())


def _score_string(submitted, expected):
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
    for task_id, answer_rec in sorted(answers.items()):
        sub = submitted.get(task_id)
        if args.mode == "rubric":
            per_task.append(
                {{
                    "task_id": task_id,
                    "status": "requires_rubric_grading",
                    "submitted": sub,
                }}
            )
            continue
        expected = _coerce_answer_value(answer_rec)
        if args.mode == "numeric":
            correct = _score_numeric(sub, expected)
        else:
            correct = _score_string(sub, expected)
        n_scored += 1
        if correct:
            n_correct += 1
        per_task.append(
            {{"task_id": task_id, "correct": bool(correct), "submitted": sub}}
        )

    summary = {{
        "n": len(answers),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "score": (n_correct / n_scored) if n_scored else 0.0,
        "per_task": per_task,
    }}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_evaluate_py(default_mode: str) -> str:
    """Return :data:`EVALUATE_PY` with ``{default_mode}`` substituted."""
    return EVALUATE_PY.format(default_mode=default_mode)


# ---------------------------------------------------------------------------
# Writers — shared by every benchmark's standardize().
# ---------------------------------------------------------------------------


def _link_problem_data(
    for_solver_dir: Path, raw_dir: Path, data_links: list[str]
) -> list[str]:
    """Create relative ``../raw/<name>`` symlinks for each problem-data name.

    Idempotent (an existing link/file at the target is unlinked first).
    Skips dot-prefixed names (VCS / HF internals) and names with no
    matching ``raw_dir`` entry. Returns the created link paths as strings.
    """
    created: list[str] = []
    for name in data_links:
        if name.startswith("."):
            continue
        if not (raw_dir / name).exists():
            continue
        link = for_solver_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "raw" / name)
        created.append(str(link))
    return created


def write_for_solver(
    *,
    for_solver_dir: Path,
    tasks: list[dict],
    raw_dir: Path,
    data_links: list[str],
) -> dict:
    """Write the agent-visible ``for_solver/`` view.

    Emits ``tasks.jsonl`` (one task per line, ``sort_keys=True``,
    ``ensure_ascii=False`` for deterministic bytes), the uniform
    ``submission.schema.json`` and a ``submission.example.json`` built
    from the first task's ``task_id`` with a placeholder answer, then
    symlinks the answer-free problem data named in ``data_links``.
    """
    for_solver_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = for_solver_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    schema_path = for_solver_dir / "submission.schema.json"
    schema_path.write_text(
        json.dumps(UNIFORM_SUBMISSION_SCHEMA, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    example_id = tasks[0]["task_id"] if tasks else "<task_id>"
    example_path = for_solver_dir / "submission.example.json"
    example_path.write_text(
        json.dumps(
            [{"task_id": example_id, "answer": "<your answer here>"}],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    symlinked = _link_problem_data(for_solver_dir, raw_dir, data_links)
    return {
        "tasks": str(tasks_path),
        "n_tasks": len(tasks),
        "schema": str(schema_path),
        "example": str(example_path),
        "symlinked": symlinked,
    }


def write_eval(
    *,
    eval_dir: Path,
    answers: list[dict],
    evaluate_py_source: str,
) -> dict:
    """Write the operator-side ``eval/`` view.

    Emits ``answers.jsonl`` (one answer per line, ``sort_keys=True``,
    ``ensure_ascii=False``) and ``evaluate.py`` from the supplied source
    string, marking the scorer executable (0o755).
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    answers_path = eval_dir / "answers.jsonl"
    with answers_path.open("w", encoding="utf-8", newline="\n") as fh:
        for answer in answers:
            fh.write(json.dumps(answer, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    evaluate_path = eval_dir / "evaluate.py"
    evaluate_path.write_text(evaluate_py_source, encoding="utf-8")
    evaluate_path.chmod(
        evaluate_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    return {
        "answers": str(answers_path),
        "n_answers": len(answers),
        "evaluate": str(evaluate_path),
    }


# Silence the unused-import linter if ``os`` is ever dropped above; kept
# imported for parity with the sibling modules' header conventions.
_ = os


__all__ = [
    "UNIFORM_SUBMISSION_SCHEMA",
    "TASK_KEYS",
    "EVALUATE_PY",
    "render_evaluate_py",
    "write_for_solver",
    "write_eval",
]

# EOF
