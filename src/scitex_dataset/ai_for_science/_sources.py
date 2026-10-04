#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_sources.py

"""Per-capsule source registration (HOST-SIDE, never solver-facing).

Records, per capsule, WHICH extracted files are the LEGITIMATE sources —
the raw problem data/code the answer derives from PLUS the computed
output produced by running the analysis — and, separately, the published
DOCS that must NOT count as a source (README / REPRODUCING / paper PDFs /
manuscript + supplement docs).

**Motivation.** Solvers cheated by grounding their answers in a capsule's
``README.md`` (which reproduces the published eval-score tables) instead
of actually running the analysis. The registry marks the legit sources
and audits the excluded docs so the operator can tell a grounded answer
from a copied one.

**Where it is emitted.** HOST-SIDE ONLY — alongside the oracle answers in
``eval/sources.jsonl``. It is DELIBERATELY kept out of the solver-facing
``for_solver`` bundle: the computed-output / results files it registers
are answer-bearing, so shipping the list would leak the very thing the
leak-strip removes.

**Classify BEFORE leak-strip.** The caller snapshots the FULL extracted
tree (data/code + results/computed-output) and classifies it *before*
:data:`._standardize.LEAK_DIRS` are stripped, so the computed output
(``results/`` etc.) is registered as a source too.

This module imports NOTHING from :mod:`._standardize`, so it can be
imported from there without a cycle. It is a strict subset of the future
full dataset-provenance artifact — the ``sources`` list maps directly
into that contract's ``sources[]`` field (name locked by scitex-clew).
"""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

# Documented doc denylist. A file is NOT a legitimate source when, on its
# case-insensitive BASENAME, any of these hold. Kept tight + easily
# extended; deliberately does NOT blanket-exclude all ``.md`` files, since
# many capsules ship legitimate ``.md`` data/notes.
DOC_SOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    # basename (lowercased) STARTS WITH any of these prefixes
    "basename_prefixes": ("readme", "reproduc"),
    # file suffix (lowercased) is any of these
    "suffixes": (".pdf",),
    # basename STEM (lowercased) CONTAINS any of these substrings
    "stem_substrings": ("manuscript", "supplement", "supplementary"),
}


def is_doc_source(relpath: str) -> bool:
    """True iff ``relpath`` is a published doc that must NOT count as a source.

    The rule is evaluated case-insensitively on the file's BASENAME
    (per :data:`DOC_SOURCE_PATTERNS`): the basename starts with ``readme``
    or ``reproduc`` (covering REPRODUCING / REPRODUCE); OR the file suffix
    is ``.pdf``; OR the basename stem contains ``manuscript``,
    ``supplement``, or ``supplementary``. Everything else is a legitimate
    source.
    """
    name = Path(relpath).name
    lname = name.lower()
    stem = Path(name).stem.lower()
    suffix = Path(name).suffix.lower()

    if lname.startswith(DOC_SOURCE_PATTERNS["basename_prefixes"]):
        return True
    if suffix in DOC_SOURCE_PATTERNS["suffixes"]:
        return True
    if any(sub in stem for sub in DOC_SOURCE_PATTERNS["stem_substrings"]):
        return True
    return False


def classify_capsule_sources(relpaths) -> dict:
    """Split ``relpaths`` into legitimate sources vs excluded docs.

    Returns ``{"sources": [...], "excluded_docs": [...]}`` with each list
    SORTED. ``excluded_docs`` are the relpaths for which
    :func:`is_doc_source` is True; ``sources`` are the rest.
    """
    sources: list[str] = []
    excluded_docs: list[str] = []
    for relpath in relpaths:
        if is_doc_source(relpath):
            excluded_docs.append(relpath)
        else:
            sources.append(relpath)
    return {"sources": sorted(sources), "excluded_docs": sorted(excluded_docs)}


def snapshot_relpaths(root: Path) -> list[str]:
    """Return SORTED POSIX relpaths of every FILE under ``root``.

    Walks ``root`` recursively and yields ``p.relative_to(root).as_posix()``
    for FILES only (directories, including empty ones, are omitted).
    """
    root = Path(root)
    relpaths = [
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    ]
    return sorted(relpaths)


def write_sources(eval_dir: Path, capsule_sources: list[dict]) -> dict:
    """Write ``eval_dir/sources.jsonl`` — one JSON object per capsule.

    Each row is serialized with ``json.dumps(row, sort_keys=True,
    ensure_ascii=False)`` + ``"\\n"``, the file opened ``newline="\\n"``.
    ``eval_dir`` is created if needed. Idempotent overwrite. Returns
    ``{"sources": str(path), "n_capsules": len(capsule_sources)}``.
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    sources_path = eval_dir / "sources.jsonl"
    with sources_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in capsule_sources:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    return {"sources": str(sources_path), "n_capsules": len(capsule_sources)}


def list_archive_members(archive_path: Path) -> list[str]:
    """Return SORTED forward-slash FILE member paths of ``archive_path``.

    Lists the archive WITHOUT extracting, so it reads the RAW (pristine,
    pre-leak-strip) capsule archive cheaply. Format handling mirrors
    :func:`._standardize._extract_archive`: ``.tar.gz`` / ``.tgz`` /
    ``.tar`` via :mod:`tarfile` (``r:*``), ``.zip`` via :mod:`zipfile`.
    Directory entries are excluded (only real files are returned).

    Raises ``FileNotFoundError`` when the archive is missing (fail loud —
    a bound capsule with no archive on disk is an operator error) and
    ``ValueError`` for an unrecognised suffix.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"capsule archive not found (or is a dangling symlink): {archive_path}"
        )
    name = archive_path.name
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            members = [m.name for m in tf.getmembers() if m.isfile()]
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
    else:
        raise ValueError(
            f"unrecognised archive suffix for {archive_path} "
            "(expected one of .tar.gz, .tgz, .tar, .zip)."
        )
    return sorted(members)


def _denest_single_top(names: list[str]) -> list[str]:
    """De-nest a single wrapping top-level dir from archive member ``names``.

    Member-list analogue of :func:`._standardize._flatten_single_top_dir`:
    if there is EXACTLY ONE distinct first path-segment AND every name
    starts with ``"<top>/"`` (a genuine wrapping dir — no top-level FILE
    equals the segment), strip the ``"<top>/"`` prefix from each name.
    Otherwise the names are returned unchanged. Result is SORTED.
    """
    if not names:
        return []
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1:
        prefix = f"{next(iter(tops))}/"
        if all(n.startswith(prefix) for n in names):
            return sorted(n[len(prefix):] for n in names)
    return sorted(names)


def register_capsule_sources(
    *,
    tasks: list[dict],
    raw_dir: Path,
    eval_dir: Path,
) -> dict:
    """Register per-capsule sources by LISTING each raw archive's members.

    The raw archives under ``raw_dir`` are the FULL, pristine,
    pre-leak-strip trees by definition — leak-stripping only ever touches
    the extracted ``for_solver/`` copy, never the raw archive. Listing
    their members therefore captures the computed output (``results/``
    etc.) as a legitimate source WITHOUT extracting anything and WITHOUT
    editing / depending on ``._standardize``'s materialization path.

    Covers EVERY capsule whose raw archive is present, regardless of the
    ``only`` selector or which capsules were materialized (raw archives
    persist) — strictly better than an only-materialized snapshot.

    Groups ``tasks`` by native capsule id, lists + de-nests each capsule's
    archive members, classifies them into sources vs excluded docs, and
    writes ``eval_dir/sources.jsonl`` via :func:`write_sources`. Capsules
    whose archive is missing on disk are skipped and reported under
    ``missing``. Returns the :func:`write_sources` result augmented with
    ``{"n_missing": int, "missing": [native_id, ...]}``.
    """
    # Lazy import: keeps this module free of a MODULE-LEVEL ._standardize
    # dependency, so ._standardize may later import from here without a
    # cycle. Internal same-package names are fine to import (read-only).
    from ._standardize import _native_capsule_id, build_capsule_index

    raw_dir = Path(raw_dir)

    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        native = _native_capsule_id(task.get("data"))
        if native is not None:
            grouped.setdefault(native, []).append(task)

    rows: list[dict] = []
    missing: list[str] = []
    for row in build_capsule_index(tasks):
        native = row["native_id"]
        archive_rel = grouped[native][0].get("data")
        if not archive_rel:
            continue
        archive_path = raw_dir / archive_rel
        if not archive_path.is_file():
            missing.append(native)
            continue
        members = _denest_single_top(list_archive_members(archive_path))
        cls = classify_capsule_sources(members)
        rows.append(
            {
                "friendly_id": row["friendly_id"],
                "native_id": native,
                "benchmark": row["benchmark"],
                "sources": cls["sources"],
                "excluded_docs": cls["excluded_docs"],
            }
        )

    wr = write_sources(eval_dir, rows)
    return {**wr, "n_missing": len(missing), "missing": sorted(missing)}


__all__ = [
    "DOC_SOURCE_PATTERNS",
    "is_doc_source",
    "classify_capsule_sources",
    "snapshot_relpaths",
    "write_sources",
    "list_archive_members",
    "register_capsule_sources",
]

# EOF
