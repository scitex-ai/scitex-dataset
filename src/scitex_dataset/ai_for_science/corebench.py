#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/corebench.py

"""CORE-Bench dataset preparation (cohort A).

Migrated from ``paper-scitex-clew/scripts/cohorts/a_corebench/dataset/``
(operator brief 2026-06-12).

CORE-Bench: 90 reproducibility-judging capsules from CodeOcean papers,
hosted off-repo at ``https://corebench.cs.princeton.edu/capsules/``.
Each capsule has 3 difficulty tiers (easy / medium / hard) — 90
underlying papers × ~3 task variants per paper.

Pipeline:

1. ``download(...)`` — pull capsule tarballs (~13 GB) from the Princeton
   CDN into ``capsule_dir``. Also fetches the operator-private answer
   manifests (``core_train.json`` + ``core_test.json``) into
   ``oracle_dir``. ``oracle_dir`` is meant to live outside any
   agent-mounted volume.
2. ``build_inventory(...)`` — read the oracle manifests, write a
   non-oracle ``inventory.json`` (task metadata only — capsule_id,
   difficulty, language, field, file counts where capsules are
   unpacked).
3. ``mask(...)`` — merge train+test, null every answer value AND the
   paper-identity fields (``capsule_title``, ``capsule_doi``); preserve
   ``capsule_id`` (opaque lookup handle), ``task_prompt``, ``field``,
   ``language``. Writes the single merged ``questions.json`` under
   ``benchmark_dir`` — exactly what the SAC capsule binds at
   ``/questions:ro``.
4. ``prepare(...)`` — runs the three above plus emits
   ``.scitex/dataset/MANIFEST.yaml`` with the snapshot id + version +
   sha256 of the masked file.

NOTE on compute: the capsule tarball download in ``download(...)`` is
~13 GB. Callers running on a SLURM cluster should ``sbatch`` it (or
call ``prepare(...)`` from a batch script) — never on a login node.
The ``mask(...)`` step is pure-Python on JSON inputs (< 1 MB total)
and safe to run anywhere.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest

# Canonical cohort identity. Matches the on-disk dir used by the paper
# repo so consumers that already point at ``data/cohort_a_corebench``
# don't need to be updated.
COHORT_ID = "corebench"
COHORT_NAME = "CORE-Bench"
COHORT_DIR = "cohort_a_corebench"
SOURCE_URL = "https://corebench.cs.princeton.edu"

# Difficulty tier labels in the order they appear in the upstream
# ``results`` arrays (per CORE-Bench paper: each capsule has
# hard/medium/easy variants; the ``results`` list is positionally
# ordered).
DIFFICULTY_TIERS = ("hard", "medium", "easy")

# Paper-identity fields nulled on every record so an agent can't web-
# search the paper for answer hints. ``capsule_id`` stays visible
# (opaque 7-digit handle the agent needs as a lookup key).
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

# Oracle source layout (matches ``paper-scitex-clew``'s
# ``download_benchmark.sh``):
#   <oracle_dir>/dataset/core_train.json   (plaintext)
#   <oracle_dir>/core_test.json            (decrypted from .gpg)
_ORACLE_TRAIN_RELPATH = ("dataset", "core_train.json")
_ORACLE_TEST_RELPATH = ("core_test.json",)


# ---------------------------------------------------------------------------
# Mask — pure-Python; works without any network and without the bulk
# capsule tarballs. This is what tests exercise.
# ---------------------------------------------------------------------------


def mask_record(rec: dict) -> dict:
    """Return a new record with answer values + paper-identity fields nulled.

    Two masks applied per record:

    1. Each element of ``rec["results"]`` is a single-key dict
       ``{question_string: answer_value}``. The mask nulls the VALUE
       only; the KEY (the question text the agent sees) is preserved.
    2. ``capsule_title`` and ``capsule_doi`` are set to ``None``
       (paper-identity mask).

    Preserved verbatim: ``field``, ``language``, ``capsule_id`` (opaque
    lookup handle), ``task_prompt``, all ``results`` question KEYS.

    Idempotent: re-masking already-masked output yields identical bytes.
    Does not mutate the input record.
    """
    out = dict(rec)
    out["results"] = [
        {q: None for q in entry} for entry in rec.get("results", [])
    ]
    for k in PAPER_IDENTITY_FIELDS:
        # Always set (even if absent in source) so the masked schema is
        # uniform across all output records.
        out[k] = None
    return out


def _load_and_mask(src: Path) -> list[dict]:
    """Read a JSON record list from ``src``, return masked records."""
    records = json.loads(src.read_text(encoding="utf-8"))
    return [mask_record(r) for r in records]


def mask(
    *,
    oracle_dir: Path,
    benchmark_dir: Path,
    **_,
) -> dict:
    """Mask the oracle JSONs and write the agent-visible ``questions.json``.

    ``oracle_dir`` must already contain the upstream-pristine
    ``dataset/core_train.json`` and ``core_test.json`` — either from a
    prior ``download(...)`` or hand-staged by the operator. The two
    record lists are concatenated (train first, then test) into a
    single 90-record ``questions.json`` under ``benchmark_dir`` —
    matches the operator brief that collapsed the older two-file split.
    """
    train = oracle_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test = oracle_dir.joinpath(*_ORACLE_TEST_RELPATH)
    missing = [str(p) for p in (train, test) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"corebench: oracle source(s) not found: {missing}. "
            "Run download(...) or stage the oracle JSONs manually."
        )

    merged: list[dict] = []
    counts: list[int] = []
    for src in (train, test):
        masked = _load_and_mask(src)
        counts.append(len(masked))
        merged.extend(masked)

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    out = benchmark_dir / "questions.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return {
        "output": str(out),
        "n_records": len(merged),
        "n_train": counts[0],
        "n_test": counts[1],
        "mask_fields": ["results[*][<question>]", *PAPER_IDENTITY_FIELDS],
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
                r.get("task_type") or r.get("difficulty") or
                (DIFFICULTY_TIERS[i] if i < len(DIFFICULTY_TIERS) else f"tier_{i}")
            )
            rows.append(
                {
                    "task_id": f"{cid}__{difficulty}",
                    "paper_id": cid,
                    "split": split,
                    "difficulty": difficulty,
                    "primary_language": lang,
                    "field": entry.get("field"),
                    "n_python_files": file_stats["n_python_files"] if file_stats else None,
                    "n_r_files": file_stats["n_r_files"] if file_stats else None,
                    "n_notebooks": file_stats["n_notebooks"] if file_stats else None,
                    "has_dockerfile": file_stats["has_dockerfile"] if file_stats else None,
                    "n_data_files": file_stats["n_data_files"] if file_stats else None,
                }
            )
    return rows


def build_inventory(
    *,
    oracle_dir: Path,
    capsule_dir: Path,
    **_,
) -> dict:
    """Write ``<capsule_dir>/../inventory.json`` from the oracle JSONs.

    The inventory contains only non-oracle metadata (no answer values),
    so it is safe to publish alongside the masked questions file. File-
    level fields (``n_python_files`` etc.) are populated only if the
    capsule has been unpacked into ``<capsule_dir>/<capsule_id>``;
    otherwise reported as ``None``.
    """
    train = oracle_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test = oracle_dir.joinpath(*_ORACLE_TEST_RELPATH)
    missing = [str(p) for p in (train, test) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"corebench: oracle source(s) not found: {missing}."
        )

    capsule_cache = capsule_dir.parent / "capsules_extracted"
    train_entries = json.loads(train.read_text(encoding="utf-8"))
    test_entries = json.loads(test.read_text(encoding="utf-8"))
    rows = (
        _build_tasks(train_entries, "train", capsule_cache)
        + _build_tasks(test_entries, "test", capsule_cache)
    )

    summary = {
        "n_capsules_total": len(train_entries) + len(test_entries),
        "n_capsules_train": len(train_entries),
        "n_capsules_test": len(test_entries),
        "n_tasks_total": len(rows),
        "n_tasks_python": sum(
            1 for r in rows if r["primary_language"] == "Python"
        ),
        "n_tasks_r": sum(1 for r in rows if r["primary_language"] == "R"),
        "by_primary_language": dict(
            Counter(r["primary_language"] for r in rows)
        ),
        "by_difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_field": dict(Counter(r["field"] for r in rows)),
        "capsule_code_in_repo": capsule_cache.exists(),
    }

    # The inventory used to live under ``src/inventory.json`` next to
    # ``src/capsules``; keep that layout.
    out = capsule_dir.parent / "inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "tasks": rows}, indent=2),
        encoding="utf-8",
    )
    return {"output": str(out), "n_tasks": len(rows), "summary": summary}


# ---------------------------------------------------------------------------
# Download — network. Pulls capsule tarballs into ``capsule_dir``. The
# oracle JSONs are NOT fetched here (they're operator-side artifacts
# the operator stages separately, by design); callers can drop them
# into ``oracle_dir`` before calling ``prepare``.
# ---------------------------------------------------------------------------


def _capsule_url(base_url: str, capsule_id: str) -> str:
    return f"{base_url}/capsules/{capsule_id}.tar.gz"


def _http_download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` via stdlib urllib.

    Streams in 1 MiB chunks so multi-MB tarballs don't peak memory.
    Caller is responsible for ``dest.parent.mkdir(exist_ok=True)``.
    """
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def download(
    *,
    oracle_dir: Path,
    capsule_dir: Path,
    capsule_ids: Iterable[str] | None = None,
    base_url: str = SOURCE_URL,
    **_,
) -> dict:
    """Pull capsule tarballs into ``capsule_dir``.

    If ``capsule_ids`` is None the inventory ``oracle_dir/dataset/core_train.json``
    + ``oracle_dir/core_test.json`` is consulted for the canonical 90-id list.
    Already-present non-empty files are skipped (idempotent — matches the
    original ``download.sh`` ``wget -c`` semantics).

    HEAVY: ~13 GB across 90 tarballs. SLURM-only on shared compute;
    never call from a login node or CI.
    """
    capsule_dir.mkdir(parents=True, exist_ok=True)

    if capsule_ids is None:
        train = oracle_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
        test = oracle_dir.joinpath(*_ORACLE_TEST_RELPATH)
        missing = [str(p) for p in (train, test) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"corebench.download: capsule_ids not given AND oracle "
                f"manifests missing: {missing}. Either pass capsule_ids "
                "explicitly or stage the oracle JSONs first."
            )
        seen: set[str] = set()
        ids: list[str] = []
        for entry in json.loads(train.read_text(encoding="utf-8")):
            cid = entry["capsule_id"]
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
        for entry in json.loads(test.read_text(encoding="utf-8")):
            cid = entry["capsule_id"]
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
        ids.sort()
        capsule_ids = ids

    n_have = n_get = n_fail = 0
    fails: list[str] = []
    for cid in capsule_ids:
        out = capsule_dir / f"{cid}.tar.gz"
        if out.exists() and out.stat().st_size > 0:
            n_have += 1
            continue
        url = _capsule_url(base_url, cid)
        try:
            _http_download(url, out)
            n_get += 1
        except Exception as exc:  # pragma: no cover — network path
            if out.exists():
                out.unlink()
            n_fail += 1
            fails.append(f"{cid}: {exc!r}")
    return {
        "capsule_dir": str(capsule_dir),
        "n_have": n_have,
        "n_fetched": n_get,
        "n_failed": n_fail,
        "failures": fails,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def prepare(
    *,
    paths: BenchmarkPaths | None = None,
    oracle_root: Path | str | None = None,
    dataset_root: Path | str | None = None,
    version: str = "v0-unstamped",
    skip_download: bool = False,
    skip_inventory: bool = False,
    **_,
) -> dict:
    """Run the full CORE-Bench preparation pipeline.

    Returns a dict summarising each step plus the path of the emitted
    ``MANIFEST.yaml``. If ``skip_download`` is True (default False), the
    capsule-tarball download step is skipped — useful when only the
    pure-Python mask path is wanted (e.g. from CI where multi-GB pulls
    are inappropriate).
    """
    if paths is None:
        paths = resolve_paths(
            COHORT_DIR, oracle_root=oracle_root, dataset_root=dataset_root
        )

    out: dict = {"cohort": COHORT_ID, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            oracle_dir=paths.oracle_dir, capsule_dir=paths.capsule_dir
        )
    if not skip_inventory:
        out["inventory"] = build_inventory(
            oracle_dir=paths.oracle_dir, capsule_dir=paths.capsule_dir
        )
    out["mask"] = mask(
        oracle_dir=paths.oracle_dir, benchmark_dir=paths.benchmark_dir
    )

    manifest_path = write_manifest(
        manifest_dir=paths.manifest_dir,
        id=COHORT_ID,
        name=COHORT_NAME,
        version=version,
        source_url=SOURCE_URL,
        cohort_dir=COHORT_DIR,
        tracked_paths=[Path(out["mask"]["output"])],
        tracked_root=paths.benchmark_dir,
        mask_seed="",  # current mask is deterministic / seed-free
    )
    out["manifest"] = str(manifest_path)
    return out


__all__ = [
    "COHORT_ID",
    "COHORT_NAME",
    "COHORT_DIR",
    "SOURCE_URL",
    "DIFFICULTY_TIERS",
    "PAPER_IDENTITY_FIELDS",
    "build_inventory",
    "download",
    "mask",
    "mask_record",
    "prepare",
]

# EOF
