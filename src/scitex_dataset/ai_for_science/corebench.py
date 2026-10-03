#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/corebench.py

"""CORE-Bench dataset preparation.

CORE-Bench: 90 reproducibility-judging capsules from CodeOcean papers,
hosted off-repo at ``https://corebench.cs.princeton.edu/capsules/``.
Each capsule can pose several questions. The ``results`` dictionaries
contain reference reruns of those questions, not difficulty variants.

Pipeline (raw → {for_solver, eval} contract — see :mod:`._base`):

1. ``download(...)`` — pull capsule tarballs (~13 GB) from the Princeton
   CDN into ``raw_dir/capsules/``, with sha256 integrity skip via
   ``raw_dir/.checksums.json``. The answer manifests
   (``dataset/core_train.json`` + ``core_test.json``) are operator-side;
   implicit capsule selection bootstraps them. Explicit ``capsule_ids``
   leaves oracle acquisition to the caller. ``raw_dir`` is never mounted.
2. ``standardize(...)`` — admit source/cache identity, then split the oracle
   into a uniform answer-masked
   per-capsule ``for_solver/`` view (no answers) and operator-side
   ``eval/answers.jsonl`` + ``eval/evaluate.py``. One question creates
    one task; all its reference samples share that task's identity.
3. ``build_inventory(...)`` — after materialization admission, read the
   oracle manifests and write ``inventory.json`` (capsule/question metadata,
   language, field and file counts) outside the selected solver capsule.
4. ``prepare(...)`` — runs the three above plus emits
   ``.scitex/dataset/MANIFEST.yaml`` with the snapshot id + version +
   checksum of the mapper. Actual source/cache identity stays private.

NOTE on compute: the capsule tarball download in ``download(...)`` is
~13 GB. Callers running on a SLURM cluster should ``sbatch`` it (or
call ``prepare(...)`` from a batch script) — never on a login node.
``standardize(...)`` uses staged JSON manifests and hashes/extracts selected
capsule archives. Budget disk space and I/O for those archives; it performs
no network acquisition itself.
"""

from __future__ import annotations

import hashlib
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
from ._sources import register_capsule_sources
from ._standardize import (
    render_evaluate_py,
    write_eval,
    write_for_solver_per_capsule,
)

# Canonical benchmark identity.
BENCHMARK = "corebench"
COHORT_ID = "corebench"
COHORT_NAME = "CORE-Bench"

# Legacy export retained for callers; reference positions never select a tier.
DIFFICULTY_TIERS = ("hard", "medium", "easy")
REFERENCE_SCHEMA = "corebench-question-references-v1"

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

# Optional pre-extracted trees for inventory file-counting live here.
# Solver materialization extracts selected archives into per-capsule input/;
# it does not bind the whole raw capsule directory.
_EXTRACTED_SUBDIR = "capsules_extracted"


# ---------------------------------------------------------------------------
# Standardize — offline JSON processing plus staged archive identity checks
# and selected input extraction. Tests use invented local archives.
# ---------------------------------------------------------------------------


def _question_task_id(capsule_id: str, question: str) -> str:
    """Stable identity for an exact, unnormalised UTF-8 question key."""
    try:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise ValueError("corebench: question key must be valid UTF-8") from None
    return f"corebench/{capsule_id}__question_{digest}"


def _split_record(rec: dict) -> tuple[list[dict], list[dict]]:
    """Create one assigned task per question and retain every scalar reference.

    Reference rows repeat the SAME task ID. Their schema/run metadata is
    evaluator-private. Missing keys are reported, never filled or discarded.
    """
    if not isinstance(rec, dict):
        raise ValueError("corebench: source record must be an object")
    cid = rec.get("capsule_id")
    if not isinstance(cid, str) or not cid or "/" in cid or "\\" in cid:
        raise ValueError("corebench: capsule_id must be a nonempty path-free string")
    if not isinstance(rec.get("task_prompt"), str):
        raise ValueError("corebench: task_prompt must be a string")
    results = rec.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("corebench: results must be a nonempty list of reference runs")
    references: dict[str, list[tuple[int, object]]] = {}
    for run_index, result_dict in enumerate(results):
        if not isinstance(result_dict, dict):
            raise ValueError(f"corebench: reference run {run_index} must be an object")
        for question, value in result_dict.items():
            if not isinstance(question, str) or not question:
                raise ValueError(f"corebench: reference run {run_index} has an invalid question key")
            references.setdefault(question, []).append((run_index, value))
    if not references:
        raise ValueError("corebench: reference runs contain no questions")
    tasks: list[dict] = []
    answers: list[dict] = []
    for question in sorted(references):
        task_id = _question_task_id(cid, question)
        samples = references[question]
        missing_runs = sorted(set(range(len(results))) - {i for i, _ in samples})
        tasks.append(
            {
                "task_id": task_id,
                "benchmark": BENCHMARK,
                "prompt": rec["task_prompt"] + "\n\nQuestion: " + question,
                "data": f"./capsules/{cid}.tar.gz",
            }
        )
        for run_index, value in samples:
            answers.append(
                {
                    "task_id": task_id,
                    "answer": {"value": value},
                    "meta": {
                        "schema": REFERENCE_SCHEMA,
                        "capsule_id": cid,
                        "question": question,
                        "reference_run_index": run_index,
                        "reference_runs_total": len(results),
                        "reference_count": len(samples),
                        "missing_reference_run_indexes": missing_runs,
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
    record lists are concatenated (train first, then test). Each question
    creates one task; each reference sample retains one private answer row.

    ``for_solver`` is written in the PER-CAPSULE shape: one self-contained
    ``capsule-NNN/`` dir per native capsule (friendly id), each holding
    the EXTRACTED capsule archive in ``input/``, a ``task.jsonl`` of only
    that capsule's rows, the uniform submission schema/example, and a
    README — plus a root ``index.jsonl`` MAPPER (friendly_id ↔ native_id).
    An agent binds exactly one ``capsule-NNN/`` dir.

    ``only`` (a friendly ``capsule-NNN`` id OR a native capsule id, e.g.
    ``capsule-0201225``) materializes just that one capsule's dir; the
    mapper remains complete. Existing capsules are reused or, with ``force``,
    re-extracted only after source/cache identity admission. Changed or
    unqualified outputs require a fresh destination.
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
    source_manifests: list[dict] = []
    for split, src in (("train", train), ("test", test)):
        source_body = src.read_bytes()
        records = json.loads(source_body)
        if not isinstance(records, list):
            raise ValueError(f"corebench: {split} source must contain a record list")
        source_manifests.append({
            "split": split, "path": src.relative_to(raw_dir).as_posix(),
            "sha256": hashlib.sha256(source_body).hexdigest(), "bytes": len(source_body),
        })
        before = len(tasks)
        for rec in records:
            rec_tasks, rec_answers = _split_record(rec)
            tasks.extend(rec_tasks)
            answers.extend(rec_answers)
        counts.append(len(tasks) - before)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("corebench: duplicate assigned question identity across records")

    # All samples remain available to the existing private leak guard.
    reference_values: dict[str, list[object]] = {}
    for answer in answers:
        reference_values.setdefault(answer["task_id"], []).append(answer["answer"]["value"])
    fs = write_for_solver_per_capsule(
        for_solver_dir=for_solver_dir,
        tasks=tasks,
        raw_dir=raw_dir,
        only=only,
        force=force,
        reference_values=reference_values,
        source_identity={
            "schema": REFERENCE_SCHEMA,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "oracle_manifests": source_manifests,
        },
    )
    ev = write_eval(
        eval_dir=eval_dir,
        answers=answers,
        evaluate_py_source=render_evaluate_py(DEFAULT_MODE),
    )
    sr = register_capsule_sources(tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir)
    return {
        "for_solver": fs,
        "eval": ev,
        "sources": sr,
        "n_tasks": len(tasks),
        "n_reference_samples": len(answers),
        "reference_schema": REFERENCE_SCHEMA,
        "n_questions_with_missing_reference_runs": len({
            a["task_id"] for a in answers if a["meta"]["missing_reference_run_indexes"]
        }),
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
        tasks, answers = _split_record(entry)
        cid = entry["capsule_id"]
        lang = entry.get("language", "Unknown")
        file_stats = _classify_capsule_files(capsule_cache / cid)
        reference_counts = Counter(answer["task_id"] for answer in answers)
        for task in tasks:
            rows.append(
                {
                    "task_id": task["task_id"],
                    "paper_id": cid,
                    "split": split,
                    "difficulty": None,
                    "reference_schema": REFERENCE_SCHEMA,
                    "reference_count": reference_counts[task["task_id"]],
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

    The inventory contains metadata without answer values and stays at the
    operator catalog root, outside the selected solver capsule bind.
    File-level fields (``n_python_files`` etc.) are
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
    if not isinstance(train_entries, list) or not isinstance(test_entries, list):
        raise ValueError("corebench: inventory sources must contain record lists")
    rows = _build_tasks(train_entries, "train", capsule_cache) + _build_tasks(
        test_entries, "test", capsule_cache
    )
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("corebench: duplicate assigned question identity in inventory")

    summary = {
        "n_capsules_total": len(train_entries) + len(test_entries),
        "n_capsules_train": len(train_entries),
        "n_capsules_test": len(test_entries),
        "n_tasks_total": len(rows),
        "n_reference_samples": sum(r["reference_count"] for r in rows),
        "n_tasks_python": sum(1 for r in rows if r["primary_language"] == "Python"),
        "n_tasks_r": sum(1 for r in rows if r["primary_language"] == "R"),
        "by_primary_language": dict(Counter(r["primary_language"] for r in rows)),
        "by_difficulty": {},  # Reference positions are not difficulty labels.
        "reference_schema": REFERENCE_SCHEMA,
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
    only: str | None = None,
    capsule_ids: Iterable[str] | None = None,
) -> dict:
    """Run the full CORE-Bench preparation pipeline.

    Returns a dict summarising each step plus the path of the emitted
    ``MANIFEST.yaml``. If ``skip_download`` is True (default False), the
    capsule-tarball download step is skipped; manifests and selected archives
    must already be staged. ``verify_integrity`` / ``force`` are passed
    to ``download`` (default: skip any capsule already on disk).
    ``capsule_ids`` restricts acquisition; ``only`` restricts materialization.
    Explicit IDs require caller-staged oracle manifests for standardization.
    """
    if paths is None:
        paths = resolve_paths(BENCHMARK, dataset_root=dataset_root)

    out: dict = {"benchmark": BENCHMARK, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            raw_dir=paths.raw_dir,
            verify_integrity=verify_integrity,
            force=force,
            capsule_ids=capsule_ids,
        )
    out["standardize"] = standardize(
        raw_dir=paths.raw_dir,
        for_solver_dir=paths.for_solver_dir,
        eval_dir=paths.eval_dir,
        force=force,
        only=only,
    )
    if not skip_inventory:
        out["inventory"] = build_inventory(
            raw_dir=paths.raw_dir, for_solver_dir=paths.for_solver_dir
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
    "REFERENCE_SCHEMA",
    "DEFAULT_MODE",
    "PAPER_IDENTITY_FIELDS",
    "build_inventory",
    "bootstrap_oracle",
    "download",
    "standardize",
    "prepare",
]

# EOF
