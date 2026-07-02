#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/corebench.py

"""CORE-Bench dataset preparation.

CORE-Bench: 90 reproducibility-judging capsules from CodeOcean papers,
hosted off-repo at ``https://corebench.cs.princeton.edu/capsules/``.
Each capsule has 3 difficulty tiers (easy / medium / hard) — 90
underlying papers × ~3 task variants per paper.

Pipeline (raw → {for_solver, eval} contract — see :mod:`._base`):

1. ``download(...)`` — pull capsule tarballs (~13 GB) from the Princeton
   CDN into ``raw_dir/capsules/``, with sha256 integrity skip via
   ``raw_dir/.checksums.json``. The answer manifests
   (``dataset/core_train.json`` + ``core_test.json``) are operator-side
   artifacts staged into ``raw_dir`` separately (not fetched here).
   ``raw_dir`` is operator-private and never mounted.
2. ``build_inventory(...)`` — read the oracle manifests, write a
   non-oracle ``inventory.json`` (task metadata only — capsule_id,
   difficulty, language, field, file counts) into the agent-visible
   ``for_solver_dir``.
3. ``standardize(...)`` — split the oracle into a uniform leak-safe
   ``for_solver/tasks.jsonl`` (no answers) + an operator-side
   ``eval/answers.jsonl`` + ``eval/evaluate.py``. Each ``results`` entry
   becomes one task; the per-tier difficulty disambiguates them. This is
   exactly what the SAC capsule binds at ``/for_solver:ro``.
4. ``prepare(...)`` — runs the three above plus emits
   ``.scitex/dataset/MANIFEST.yaml`` with the snapshot id + version +
   sha256 of the tasks file.

NOTE on compute: the capsule tarball download in ``download(...)`` is
~13 GB. Callers running on a SLURM cluster should ``sbatch`` it (or
call ``prepare(...)`` from a batch script) — never on a login node.
The ``standardize(...)`` step is pure-Python on JSON inputs (< 1 MB
total) and safe to run anywhere.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from ._base import BenchmarkPaths, resolve_paths
from ._corebench_download import (
    SOURCE_URL,
    _ORACLE_TEST_RELPATH,
    _ORACLE_TRAIN_RELPATH,
    bootstrap_oracle,
    download,
)
from ._manifest import write_manifest
from ._standardize import (
    render_evaluate_py,
    write_eval,
    write_for_solver_per_capsule,
)

# Canonical benchmark identity.
BENCHMARK = "corebench"
COHORT_ID = "corebench"
COHORT_NAME = "CORE-Bench"

# Difficulty tier labels in the order they appear in the upstream
# ``results`` arrays (per CORE-Bench paper: each capsule has
# hard/medium/easy variants; the ``results`` list is positionally
# ordered).
DIFFICULTY_TIERS = ("hard", "medium", "easy")

# Default scorer mode baked into eval/evaluate.py — CORE-Bench answers
# are numeric report values, scored within relative tolerance.
DEFAULT_MODE = "numeric"

# Paper-identity fields dropped from the leak-safe task view so an agent
# can't web-search the paper for answer hints. ``capsule_id`` stays
# visible (opaque 7-digit handle the agent needs as a lookup key).
PAPER_IDENTITY_FIELDS = ("capsule_title", "capsule_doi")

# File-extension buckets for inventory file-counting.
_PY_EXTS = (".py",)
_R_EXTS = (".r", ".rmd")
_IPYNB_EXTS = (".ipynb",)
_DATA_EXTS = (
    ".csv",
    ".tsv",
    ".npy",
    ".h5",
    ".parquet",
    ".rds",
    ".rdata",
    ".mat",
    ".json",
    ".xlsx",
)

# Oracle source layout on disk — bootstrapped into raw_dir by
# ``download(...)`` (see :mod:`._corebench_download`). The relpaths
# ``_ORACLE_TRAIN_RELPATH`` / ``_ORACLE_TEST_RELPATH`` are imported from
# that module; the readers below consume them:
#   <raw_dir>/dataset/core_train.json   (plaintext)
#   <raw_dir>/core_test.json            (decrypted from .gpg)

# Capsule tarballs are the only raw content symlinked into for_solver/;
# the whole ``raw/capsules`` dir is linked once (not per-task) and each
# task's ``data`` points inside it. Extracted trees (for inventory
# file-counting) land under this subdir.
_EXTRACTED_SUBDIR = "capsules_extracted"


# ---------------------------------------------------------------------------
# Standardize — pure-Python; works without any network and without the
# bulk capsule tarballs. This is what tests exercise.
# ---------------------------------------------------------------------------


def _split_record(rec: dict) -> tuple[list[dict], list[dict]]:
    """Split one oracle record into (tasks, answers) row lists.

    Each ``rec["results"]`` entry is a ``{question_text: answer_value}``
    dict for one difficulty tier — and may hold MORE THAN ONE question
    (CoreBench tasks pose 1–8 questions per tier). Every question becomes
    its own standardized task so each answer stays a scalar (matching the
    per-question metric): ``corebench/<cid>__<difficulty>__q<j>``. The
    positional index selects the tier (hard/medium/easy, then
    ``tier_<i>``). Tasks carry the uniform ``{task_id, benchmark, prompt,
    data}`` keys only — no answer-bearing fields. Answers carry the
    matching ``task_id``, the ``{"value": ...}`` payload, and meta.
    """
    cid = rec["capsule_id"]
    tasks: list[dict] = []
    answers: list[dict] = []
    for i, result_dict in enumerate(rec.get("results", [])):
        difficulty = DIFFICULTY_TIERS[i] if i < 3 else f"tier_{i}"
        for j, (question_text, answer_value) in enumerate(result_dict.items()):
            task_id = f"corebench/{cid}__{difficulty}__q{j}"
            tasks.append(
                {
                    "task_id": task_id,
                    "benchmark": BENCHMARK,
                    "prompt": rec["task_prompt"] + "\n\nQuestion: " + question_text,
                    "data": f"./capsules/{cid}.tar.gz",
                }
            )
            answers.append(
                {
                    "task_id": task_id,
                    "answer": {"value": answer_value},
                    "meta": {
                        "capsule_id": cid,
                        "difficulty": difficulty,
                        "question": question_text,
                        "field": rec.get("field"),
                        "language": rec.get("language"),
                    },
                }
            )
    return tasks, answers


def standardize(
    *,
    raw_dir: Path,
    for_solver_dir: Path,
    eval_dir: Path,
    only: str | None = None,
    force: bool = False,
    **_,
) -> dict:
    """Split the oracle JSONs into the for_solver + eval views.

    ``raw_dir`` must already contain the upstream-pristine
    ``dataset/core_train.json`` and ``core_test.json`` — either from a
    prior ``download(...)`` or hand-staged by the operator. The two
    record lists are concatenated (train first, then test); each
    ``results`` entry becomes one task (leak-safe) + one answer.

    ``for_solver`` is written in the PER-CAPSULE shape: one self-contained
    ``capsule-NNN/`` dir per native capsule (friendly id), each holding
    the EXTRACTED capsule archive in ``input/``, a ``task.jsonl`` of only
    that capsule's rows, the uniform submission schema/example, and a
    README — plus a root ``index.jsonl`` MAPPER (friendly_id ↔ native_id).
    An agent binds exactly one ``capsule-NNN/`` dir.

    ``only`` (a friendly ``capsule-NNN`` id OR a native capsule id, e.g.
    ``capsule-0201225``) materializes just that one capsule's dir; the
    mapper is always written in full. ``force`` re-extracts capsules that
    are already present (default skips them).
    """
    train = raw_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test = raw_dir.joinpath(*_ORACLE_TEST_RELPATH)
    missing = [str(p) for p in (train, test) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"corebench: oracle source(s) not found: {missing}. "
            "Run `corebench download` first — it bootstraps the oracle "
            "manifests (fetch + gpg-decrypt) into raw/."
        )

    tasks: list[dict] = []
    answers: list[dict] = []
    counts: list[int] = []
    for src in (train, test):
        records = json.loads(src.read_text(encoding="utf-8"))
        before = len(tasks)
        for rec in records:
            rec_tasks, rec_answers = _split_record(rec)
            tasks.extend(rec_tasks)
            answers.extend(rec_answers)
        counts.append(len(tasks) - before)

    answer_values = {
        a["task_id"]: a["answer"]["value"]
        for a in answers
        if isinstance(a.get("answer"), dict) and "value" in a["answer"]
    }
    fs = write_for_solver_per_capsule(
        for_solver_dir=for_solver_dir,
        tasks=tasks,
        raw_dir=raw_dir,
        only=only,
        force=force,
        answer_values=answer_values,
    )
    ev = write_eval(
        eval_dir=eval_dir,
        answers=answers,
        evaluate_py_source=render_evaluate_py(DEFAULT_MODE),
    )
    return {
        "for_solver": fs,
        "eval": ev,
        "n_tasks": len(tasks),
        "n_train": counts[0],
        "n_test": counts[1],
        "default_mode": DEFAULT_MODE,
    }


# ---------------------------------------------------------------------------
# Inventory — also pure-Python (no network). Walks any extracted
# capsule trees that happen to be present and writes ``inventory.json``.
# ---------------------------------------------------------------------------


def _classify_capsule_files(capsule_dir: Path) -> Optional[dict]:
    """Walk a downloaded capsule directory and count files by extension."""
    if not capsule_dir.exists():
        return None
    n_py = n_r = n_ipynb = n_data = 0
    has_dockerfile = False
    for _root, _dirs, files in os.walk(capsule_dir):
        for f in files:
            fl = f.lower()
            if fl.endswith(_PY_EXTS):
                n_py += 1
            elif fl.endswith(_R_EXTS):
                n_r += 1
            elif fl.endswith(_IPYNB_EXTS):
                n_ipynb += 1
            elif fl in ("dockerfile",) or fl.startswith("dockerfile"):
                has_dockerfile = True
            elif fl.endswith(_DATA_EXTS):
                n_data += 1
    return {
        "n_python_files": n_py,
        "n_r_files": n_r,
        "n_notebooks": n_ipynb,
        "has_dockerfile": has_dockerfile,
        "n_data_files": n_data,
    }


def _build_tasks(
    entries: Iterable[dict], split: str, capsule_cache: Path
) -> list[dict]:
    rows: list[dict] = []
    for entry in entries:
        cid = entry["capsule_id"]
        lang = entry.get("language", "Unknown")
        file_stats = _classify_capsule_files(capsule_cache / cid)
        results = entry.get("results", [])
        for i, r in enumerate(results):
            difficulty = (
                r.get("task_type")
                or r.get("difficulty")
                or (DIFFICULTY_TIERS[i] if i < len(DIFFICULTY_TIERS) else f"tier_{i}")
            )
            rows.append(
                {
                    "task_id": f"{cid}__{difficulty}",
                    "paper_id": cid,
                    "split": split,
                    "difficulty": difficulty,
                    "primary_language": lang,
                    "field": entry.get("field"),
                    "n_python_files": file_stats["n_python_files"]
                    if file_stats
                    else None,
                    "n_r_files": file_stats["n_r_files"] if file_stats else None,
                    "n_notebooks": file_stats["n_notebooks"] if file_stats else None,
                    "has_dockerfile": file_stats["has_dockerfile"]
                    if file_stats
                    else None,
                    "n_data_files": file_stats["n_data_files"] if file_stats else None,
                }
            )
    return rows


def build_inventory(
    *,
    raw_dir: Path,
    for_solver_dir: Path,
    **_,
) -> dict:
    """Write ``for_solver_dir/inventory.json`` from the oracle JSONs.

    The inventory contains only non-oracle metadata (no answer values),
    so it is published in the agent-visible ``for_solver_dir`` alongside
    the tasks file. File-level fields (``n_python_files`` etc.) are
    populated only if the capsule has been unpacked into
    ``raw_dir/capsules_extracted/<capsule_id>``; otherwise ``None``.
    """
    train = raw_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test = raw_dir.joinpath(*_ORACLE_TEST_RELPATH)
    missing = [str(p) for p in (train, test) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"corebench: oracle source(s) not found: {missing}. "
            "Run `corebench download` first — it bootstraps the oracle "
            "manifests (fetch + gpg-decrypt) into raw/."
        )

    capsule_cache = raw_dir / _EXTRACTED_SUBDIR
    train_entries = json.loads(train.read_text(encoding="utf-8"))
    test_entries = json.loads(test.read_text(encoding="utf-8"))
    rows = _build_tasks(train_entries, "train", capsule_cache) + _build_tasks(
        test_entries, "test", capsule_cache
    )

    summary = {
        "n_capsules_total": len(train_entries) + len(test_entries),
        "n_capsules_train": len(train_entries),
        "n_capsules_test": len(test_entries),
        "n_tasks_total": len(rows),
        "n_tasks_python": sum(1 for r in rows if r["primary_language"] == "Python"),
        "n_tasks_r": sum(1 for r in rows if r["primary_language"] == "R"),
        "by_primary_language": dict(Counter(r["primary_language"] for r in rows)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_field": dict(Counter(r["field"] for r in rows)),
        "capsule_code_in_repo": capsule_cache.exists(),
    }

    for_solver_dir.mkdir(parents=True, exist_ok=True)
    out = for_solver_dir / "inventory.json"
    out.write_text(
        json.dumps({"summary": summary, "tasks": rows}, indent=2),
        encoding="utf-8",
    )
    return {"output": str(out), "n_tasks": len(rows), "summary": summary}


# ---------------------------------------------------------------------------
# Download — network. The oracle bootstrap + capsule-tarball fetch live in
# :mod:`._corebench_download` (kept there for the 512-line-per-file guard);
# ``download`` and ``bootstrap_oracle`` are re-exported at module top so
# ``corebench.download`` / ``corebench.bootstrap_oracle`` stay callable.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def prepare(
    *,
    paths: BenchmarkPaths | None = None,
    dataset_root: Path | str | None = None,
    version: str = "v0-unstamped",
    skip_download: bool = False,
    skip_inventory: bool = False,
    verify_integrity: bool = False,
    force: bool = False,
    **_,
) -> dict:
    """Run the full CORE-Bench preparation pipeline.

    Returns a dict summarising each step plus the path of the emitted
    ``MANIFEST.yaml``. If ``skip_download`` is True (default False), the
    capsule-tarball download step is skipped — useful when only the
    pure-Python standardize path is wanted (e.g. from CI where multi-GB
    pulls are inappropriate). ``verify_integrity`` / ``force`` are passed
    to ``download`` (default: skip any capsule already on disk).
    """
    if paths is None:
        paths = resolve_paths(BENCHMARK, dataset_root=dataset_root)

    out: dict = {"benchmark": BENCHMARK, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            raw_dir=paths.raw_dir,
            verify_integrity=verify_integrity,
            force=force,
        )
    if not skip_inventory:
        out["inventory"] = build_inventory(
            raw_dir=paths.raw_dir, for_solver_dir=paths.for_solver_dir
        )
    out["standardize"] = standardize(
        raw_dir=paths.raw_dir,
        for_solver_dir=paths.for_solver_dir,
        eval_dir=paths.eval_dir,
        force=force,
    )

    manifest_path = write_manifest(
        manifest_dir=paths.manifest_dir,
        id=COHORT_ID,
        name=COHORT_NAME,
        version=version,
        source_url=SOURCE_URL,
        benchmark=BENCHMARK,
        tracked_paths=[Path(out["standardize"]["for_solver"]["index"])],
        tracked_root=paths.for_solver_dir,
        mask_seed="",  # standardize is deterministic / seed-free
    )
    out["manifest"] = str(manifest_path)
    return out


__all__ = [
    "BENCHMARK",
    "COHORT_ID",
    "COHORT_NAME",
    "SOURCE_URL",
    "DIFFICULTY_TIERS",
    "DEFAULT_MODE",
    "PAPER_IDENTITY_FIELDS",
    "build_inventory",
    "bootstrap_oracle",
    "download",
    "standardize",
    "prepare",
]

# EOF
