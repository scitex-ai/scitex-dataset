#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_standardize.py

"""Shared ``standardize`` helpers — raw oracle → {for_solver, eval}.

Every AI-for-science benchmark splits its operator-private ``raw/``
oracle (which *has* answers) into two derived views with a UNIFORM
on-disk contract, so the experiment harness treats all benchmarks
identically:

- ``for_solver/`` — the AGENT view (leak-safe). Two shapes are
  supported:

  * the FLAT shape (:func:`write_for_solver`) — a global ``tasks.jsonl``
    of ``{task_id, benchmark, prompt, data}`` records (NO answers), a
    ``submission.schema.json``, a one-row ``submission.example.json``,
    and relative symlinks to the answer-free problem data under
    ``raw/``;
  * the PER-CAPSULE shape (:func:`write_for_solver_per_capsule`) — one
    self-contained ``capsule-NNN/`` dir per native capsule (FRIENDLY id,
    human-communicable) holding the EXTRACTED problem archive in
    ``input/``, a ``task.jsonl`` of ONLY that capsule's rows (each
    ``data`` rewritten to ``./input``), a copied
    ``submission.schema.json``, a ``submission.example.json`` pre-filled
    with this capsule's real ``task_id``(s), and a plain-language
    ``README.md``. A root ``index.jsonl`` MAPPER (operator-facing, never
    mounted) records every friendly_id → native_id assignment. An agent
    binds EXACTLY one ``capsule-NNN/`` dir so no sibling capsule (and no
    answer) is ever reachable.
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
import tarfile
import zipfile
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


# ---------------------------------------------------------------------------
# Per-capsule materializer — one self-contained, friendly-id'd dir per
# native capsule, plus a root index.jsonl MAPPER. This is the shape an
# agent binds (EXACTLY one capsule-NNN/ dir, never the whole tree).
# ---------------------------------------------------------------------------

# Archive suffixes recognised for extraction, longest first so the
# compound ``.tar.gz`` wins over a bare ``.gz`` when stripping the id.
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

# Files copied/written into every per-capsule dir.
_TASK_FILENAME = "task.jsonl"
_INPUT_SUBDIR = "input"
_INDEX_FILENAME = "index.jsonl"
_README_FILENAME = "README.md"
_SCHEMA_FILENAME = "submission.schema.json"
_EXAMPLE_FILENAME = "submission.example.json"

# Top-level subdirectories inside an extracted capsule that ship the
# authors' ORIGINAL outputs — i.e. the answer the task asks the agent to
# reproduce (e.g. CoreBench's ``results/classification/log_evaluate.txt``
# holds the eval-loss). They are stripped from the agent-visible
# ``input/`` after extraction so the answer is never reachable; the
# canonical answers live only in the operator-side ``eval/answers.jsonl``
# (never mounted). Listed as a constant so the leak surface is auditable
# and extensible per benchmark.
LEAK_DIRS = ("results",)

# Per-capsule friendly-id directory prefix (``capsule-NNN``). The
# ``for_solver/`` root is reduced to ONLY these dirs + ``index.jsonl`` by
# :func:`_purge_legacy_root_artifacts`, so ANY other root entry is removed
# regardless of its name — the old flat layout's global ``tasks.jsonl`` /
# ``submission.*`` AND the benchmark-specific problem-data symlinks the old
# :func:`write_for_solver` created (CoreBench ``capsules``, BixBench
# ``CapsuleFolder-*.zip``, BioMysteryBench ``data``), each of which would
# otherwise re-expose every capsule/task and break the per-capsule contract.
_CAPSULE_DIR_PREFIX = "capsule-"


def friendly_capsule_id(index: int) -> str:
    """Return the deterministic friendly dir name for a 0-based position.

    Friendly ids are ``capsule-NNN`` zero-padded to 3 digits and 1-based
    (``index=0`` → ``capsule-001``), assigned over the native capsule ids
    sorted ascending PER BENCHMARK.
    """
    return f"capsule-{index + 1:03d}"


def _native_capsule_id(data: str | None) -> str | None:
    """Derive the native capsule id from a task's ``data`` archive path.

    ``data`` points at the capsule's archive, e.g.
    ``"./capsules/capsule-0201225.tar.gz"`` →  ``"capsule-0201225"``.
    The leading directory and any recognised archive suffix are stripped.
    Returns ``None`` when ``data`` is falsy (task has no bound archive).
    """
    if not data:
        return None
    name = Path(data).name
    for suffix in _ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_capsule_index(tasks: list[dict]) -> list[dict]:
    """Group ``tasks`` by native capsule and assign friendly ids.

    Tasks are grouped by the native capsule id derived from each row's
    ``data`` archive path; the distinct native ids are sorted ascending
    and given ``capsule-NNN`` friendly ids in that order. Returns one
    mapper row per capsule, in friendly-id order::

        {"friendly_id", "native_id", "benchmark", "task_ids", "dir"}

    This is computed from the FULL task list (independent of any ``only``
    filter) so the mapper is always complete. Tasks whose ``data`` yields
    no capsule id are skipped (they have nothing to materialize).
    """
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        native = _native_capsule_id(task.get("data"))
        if native is None:
            continue
        grouped.setdefault(native, []).append(task)

    index: list[dict] = []
    for position, native in enumerate(sorted(grouped)):
        friendly = friendly_capsule_id(position)
        rows = grouped[native]
        index.append(
            {
                "friendly_id": friendly,
                "native_id": native,
                "benchmark": rows[0].get("benchmark"),
                "task_ids": [r["task_id"] for r in rows],
                "dir": friendly,
            }
        )
    return index


def _resolve_only(only: str, index: list[dict]) -> dict:
    """Resolve an ``--only`` selector (friendly OR native id) to its row.

    Raises ``KeyError`` if it matches neither a ``friendly_id`` nor a
    ``native_id`` in the mapper.
    """
    for row in index:
        if only in (row["friendly_id"], row["native_id"]):
            return row
    raise KeyError(
        f"--only {only!r} matched no capsule (neither a friendly_id "
        f"capsule-NNN nor a native capsule id) in the {len(index)}-capsule index."
    )


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract ``archive_path`` into ``dest_dir`` (stdlib, by extension).

    ``.tar.gz`` / ``.tgz`` / ``.tar`` go through :mod:`tarfile`; ``.zip``
    through :mod:`zipfile`. ``dest_dir`` is created fresh by the caller.
    Raises ``FileNotFoundError`` if the archive is missing (fail loud:
    a selected capsule with no archive on disk is an operator error),
    and ``ValueError`` for an unrecognised suffix.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"capsule archive not found (or is a dangling symlink): {archive_path}"
        )
    name = archive_path.name
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(
            f"unrecognised archive suffix for {archive_path} "
            f"(expected one of {_ARCHIVE_SUFFIXES})."
        )


def _flatten_single_top_dir(input_dir: Path) -> None:
    """De-nest ``input_dir`` when the archive wrapped everything in one dir.

    Many capsule tarballs (e.g. CoreBench) contain a SINGLE top-level
    directory named after the capsule (``capsule-0220918/``), so a naive
    extract yields ``input/capsule-0220918/code…`` instead of
    ``input/code…`` — and the task's ``data: "./input"`` pointer then
    misses by one level. If ``input_dir`` holds EXACTLY one entry and it
    is a directory, this lifts that directory's children up into
    ``input_dir`` and removes the now-empty wrapper. The general case is
    handled conservatively: a multi-entry archive (or a lone top-level
    FILE) is left untouched.
    """
    entries = list(input_dir.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    wrapper = entries[0]
    # Two-phase to dodge a name collision with the wrapper itself: stage
    # to a sibling temp dir, then move children up and drop the wrapper.
    staging = input_dir.parent / f"_flatten_{input_dir.name}"
    if staging.exists():
        _rmtree(staging)
    wrapper.rename(staging)
    for child in staging.iterdir():
        child.rename(input_dir / child.name)
    staging.rmdir()


def _strip_leak_dirs(input_dir: Path) -> list[str]:
    """Remove answer-bearing top-level dirs (:data:`LEAK_DIRS`) from ``input_dir``.

    For each name in :data:`LEAK_DIRS`, if ``input_dir/<name>`` exists as a
    directory it is recursively deleted so the authors' original outputs
    (the answer the task asks to reproduce) cannot be read by the agent.
    Only TOP-LEVEL entries are stripped; nested non-leak data is kept.
    Returns the removed paths as strings (for the caller's report / tests).
    """
    removed: list[str] = []
    for name in LEAK_DIRS:
        leak = input_dir / name
        if leak.is_dir() and not leak.is_symlink():
            _rmtree(leak)
            removed.append(str(leak))
    return removed


def _strip_notebook_outputs(input_dir: Path) -> list[str]:
    """Clear executed-output cells from every ``*.ipynb`` under ``input_dir``.

    A capsule may ship an EXECUTED notebook (e.g. BixBench's
    ``CapsuleNotebook-*/..._executed.ipynb``) whose code-cell OUTPUTS are the
    very results the task asks the agent to reproduce — an answer leak that
    the whole-dir :func:`_strip_leak_dirs` pass does not catch. This walks
    every notebook under the agent-visible ``input/`` and clears each cell's
    ``outputs`` + ``execution_count`` (nbstripout-style), keeping the notebook
    CODE as a scaffold while removing the computed answers. A no-op for
    capsules without notebooks (CoreBench, BioMysteryBench). The canonical
    answers live only in operator-side ``eval/answers.jsonl`` (never mounted).
    Returns the cleared notebook paths as strings.
    """
    cleared: list[str] = []
    for nb_path in sorted(input_dir.rglob("*.ipynb")):
        if not nb_path.is_file():
            continue
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
        if changed:
            nb_path.write_text(
                json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            cleared.append(str(nb_path))
    return cleared


def _purge_legacy_root_artifacts(for_solver_dir: Path) -> list[str]:
    """Reduce the ``for_solver/`` root to ONLY ``capsule-NNN/`` dirs + the mapper.

    The per-capsule contract is that ``for_solver/`` holds nothing but the
    per-capsule ``capsule-NNN/`` directories and ``index.jsonl``. Every other
    root entry re-exposes capsules/tasks beyond the single capsule an agent
    binds, so it is removed regardless of name:

    - the old flat layout's global ``tasks.jsonl`` / ``submission.*``;
    - benchmark-specific problem-data symlinks the old
      :func:`write_for_solver` created at the root (CoreBench ``capsules``,
      BixBench ``CapsuleFolder-*.zip``, BioMysteryBench ``data``).

    A ``capsule-NNN`` REAL directory (friendly id — prefix + all-digit
    suffix) and ``index.jsonl`` are always preserved. Returns the removed
    paths as strings.
    """
    if not for_solver_dir.is_dir():
        return []
    removed: list[str] = []
    for entry in sorted(for_solver_dir.iterdir()):
        if entry.name == _INDEX_FILENAME:
            continue
        is_capsule_dir = (
            entry.is_dir()
            and not entry.is_symlink()
            and entry.name.startswith(_CAPSULE_DIR_PREFIX)
            and entry.name[len(_CAPSULE_DIR_PREFIX) :].isdigit()
        )
        if is_capsule_dir:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            _rmtree(entry)
        removed.append(str(entry))
    return removed


def _render_capsule_readme(
    *, friendly_id: str, benchmark: str | None, task_ids: list[str]
) -> str:
    """Return plain-language submission instructions for one capsule.

    Names the capsule by its FRIENDLY id, lists the exact ``task_id``(s)
    to answer, states the uniform JSON-array submission contract, and
    points at ``submission.example.json`` + the extracted ``input/`` dir.
    """
    bench = benchmark or "this benchmark"
    id_lines = "\n".join(f"- `{tid}`" for tid in task_ids)
    return f"""# {friendly_id}

This directory is one self-contained {bench} task capsule. Everything
you need is inside it; no other capsule is referenced.

## What to do

1. The problem's code and data are EXTRACTED under `{_INPUT_SUBDIR}/`.
   Read `{_INPUT_SUBDIR}/` and work the task described in `{_TASK_FILENAME}`.
2. Answer the following task id(s):

{id_lines}

## How to submit

Write a single JSON file: an ARRAY of objects, one per task id above,
each shaped `{{"task_id": "<id>", "answer": <your answer>}}` — exactly
the contract in `{_SCHEMA_FILENAME}`. A ready-to-edit template with the
real task id(s) filled in is in `{_EXAMPLE_FILENAME}`; replace each
`"<your answer here>"` with your answer and keep the `task_id` values
unchanged.
"""


def write_for_solver_per_capsule(
    *,
    for_solver_dir: Path,
    tasks: list[dict],
    raw_dir: Path,
    only: str | None = None,
    force: bool = False,
) -> dict:
    """Write the per-capsule, friendly-id'd ``for_solver/`` view + mapper.

    Groups ``tasks`` by native capsule (via each row's ``data`` archive
    path), assigns deterministic ``capsule-NNN`` friendly ids over the
    sorted native ids, and writes the operator-facing mapper
    ``for_solver/index.jsonl`` (one row per capsule — ALWAYS the full set,
    regardless of ``only``). Then, for each materialized capsule, creates
    a self-contained ``for_solver/capsule-NNN/`` dir holding:

    - ``input/``                  — the raw archive EXTRACTED here, then
                                de-nested if it wrapped everything in a
                                single top-level dir (so ``input/`` holds
                                ``code/``, ``data/``, … directly), and with
                                the answer-bearing :data:`LEAK_DIRS` (e.g.
                                ``results/``) stripped so the agent cannot
                                read the authors' original outputs;
    - ``task.jsonl``              — ONLY this capsule's task rows, each
                                row's ``data`` rewritten to ``"./input"``;
    - ``submission.schema.json``  — the uniform submission schema;
    - ``submission.example.json`` — pre-filled with THIS capsule's real
                                task_id(s) + placeholder answers;
    - ``README.md``               — plain-language instructions.

    Before anything is written, :func:`_purge_legacy_root_artifacts`
    reduces the ``for_solver/`` root to ONLY ``capsule-NNN/`` dirs +
    ``index.jsonl`` — removing any old flat-layout artifact (global
    ``tasks.jsonl`` / ``submission.*``) or benchmark-specific problem-data
    symlink (CoreBench ``capsules``, BixBench ``CapsuleFolder-*.zip``,
    BioMysteryBench ``data``) that would otherwise re-expose every capsule.

    ``only`` (friendly id ``capsule-NNN`` OR a native capsule id)
    restricts materialization to a single capsule's dir; the mapper is
    still written in full. Default materializes every capsule.

    Idempotent: an existing ``capsule-NNN/input`` is skipped unless
    ``force=True``, which removes and re-extracts the capsule dir.

    ``raw_dir`` is the operator-private snapshot root; each capsule's
    archive is resolved as ``raw_dir / <data-relative-to-for_solver>``.
    """
    for_solver_dir.mkdir(parents=True, exist_ok=True)

    # Purge any leftover root artifacts from the old flat layout so the
    # tree obeys the per-capsule contract (only capsule-*/ + index.jsonl).
    # Runs every standardize, independent of ``only``.
    purged_legacy = _purge_legacy_root_artifacts(for_solver_dir)

    index = build_capsule_index(tasks)
    index_path = for_solver_dir / _INDEX_FILENAME
    with index_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in index:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    if only is not None:
        selected = [_resolve_only(only, index)]
    else:
        selected = index

    # Re-group tasks by native id once so each capsule writes only its own.
    by_native: dict[str, list[dict]] = {}
    for task in tasks:
        native = _native_capsule_id(task.get("data"))
        if native is not None:
            by_native.setdefault(native, []).append(task)

    materialized: list[str] = []
    skipped: list[str] = []
    stripped_leaks: list[str] = []
    cleared_notebooks: list[str] = []
    for row in selected:
        native = row["native_id"]
        capsule_dir = for_solver_dir / row["dir"]
        input_dir = capsule_dir / _INPUT_SUBDIR
        rows = by_native[native]

        # The archive path is shared across this capsule's rows.
        archive_rel = rows[0]["data"]
        archive_path = (raw_dir / archive_rel).resolve()

        if input_dir.exists() and not force:
            skipped.append(str(capsule_dir))
            continue
        if force and input_dir.exists():
            _rmtree(input_dir)

        capsule_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, input_dir)
        # De-nest a single wrapping top-level dir so input/ holds code/,
        # data/, … directly, making the ``data: "./input"`` pointer exact.
        _flatten_single_top_dir(input_dir)
        # Strip the answer-leak AFTER flattening so it catches input/results/
        # (the authors' original outputs) regardless of the archive shape.
        stripped_leaks.extend(_strip_leak_dirs(input_dir))
        # Clear executed-notebook output cells (e.g. BixBench's
        # ``*_executed.ipynb`` leaks the answer in its outputs) — keeps the
        # notebook CODE as scaffold, removes the computed answers. No-op for
        # capsules without notebooks.
        cleared_notebooks.extend(_strip_notebook_outputs(input_dir))

        # task.jsonl — this capsule's rows only, data rewritten to ./input.
        capsule_tasks = [{**task, "data": f"./{_INPUT_SUBDIR}"} for task in rows]
        task_path = capsule_dir / _TASK_FILENAME
        with task_path.open("w", encoding="utf-8", newline="\n") as fh:
            for task in capsule_tasks:
                fh.write(json.dumps(task, sort_keys=True, ensure_ascii=False))
                fh.write("\n")

        # Uniform schema, copied into every capsule dir.
        (capsule_dir / _SCHEMA_FILENAME).write_text(
            json.dumps(UNIFORM_SUBMISSION_SCHEMA, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Example pre-filled with this capsule's real task_id(s).
        (capsule_dir / _EXAMPLE_FILENAME).write_text(
            json.dumps(
                [
                    {"task_id": tid, "answer": "<your answer here>"}
                    for tid in row["task_ids"]
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # Plain-language instructions, friendly id in the human-facing text.
        (capsule_dir / _README_FILENAME).write_text(
            _render_capsule_readme(
                friendly_id=row["friendly_id"],
                benchmark=row["benchmark"],
                task_ids=row["task_ids"],
            ),
            encoding="utf-8",
        )
        materialized.append(str(capsule_dir))

    return {
        "index": str(index_path),
        "n_capsules": len(index),
        "n_materialized": len(materialized),
        "n_skipped": len(skipped),
        "materialized": materialized,
        "skipped": skipped,
        "stripped_leaks": stripped_leaks,
        "cleared_notebooks": cleared_notebooks,
        "purged_legacy": purged_legacy,
        "only": only,
    }


def _rmtree(path: Path) -> None:
    """Recursively remove a directory tree (stdlib ``shutil`` lazy import)."""
    import shutil

    shutil.rmtree(path)


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
    "LEAK_DIRS",
    "EVALUATE_PY",
    "render_evaluate_py",
    "write_for_solver",
    "write_for_solver_per_capsule",
    "build_capsule_index",
    "friendly_capsule_id",
    "write_eval",
]

# EOF
