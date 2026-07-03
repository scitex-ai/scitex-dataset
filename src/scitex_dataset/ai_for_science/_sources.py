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


__all__ = [
    "DOC_SOURCE_PATTERNS",
    "is_doc_source",
    "classify_capsule_sources",
    "snapshot_relpaths",
    "write_sources",
]

# EOF
