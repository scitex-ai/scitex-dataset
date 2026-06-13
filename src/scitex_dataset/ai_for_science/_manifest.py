#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_manifest.py

"""Provenance ``MANIFEST.yaml`` emission for agentic-benchmark cohorts.

Mirrors scitex-template's ``.scitex/template/MANIFEST.yaml`` convention
(``feat/clone-provenance-manifest``): a downstream consumer can pin a
specific dataset snapshot by id + version + checksum + mask-seed so the
clew reproducibility verifier rejects drifted inputs.

Schema (one document per cohort, written deterministically as YAML):

    id:          str   # canonical cohort id, e.g. "corebench"
    name:        str   # human label, e.g. "CORE-Bench"
    version:     str   # snapshot tag — upstream tag, HF revision, or
                       # ISO-date for non-versioned upstreams
    source_url:  str   # canonical upstream URL
    benchmark:   str   # canonical benchmark name (on-disk dir under ai-for-science/)
    mask_seed:   str   # deterministic seed for the mask transform
                       # (empty = no randomness; current masks all use ``""``)
    files:       list  # [{path: <rel>, sha256: <hex>, size: <int>}, ...]
    scitex_dataset_version: str
    prepared_at: str   # ISO-8601 UTC timestamp

Writing is YAML-via-stdlib: a small handwritten emitter is used so we
don't take a PyYAML dependency just for one file. The output is
``sort_keys`` deterministic — re-running ``prepare`` against an
unchanged snapshot produces identical bytes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MANIFEST_FILENAME = "MANIFEST.yaml"

# Chunk size for streaming sha256 — large enough to avoid syscall churn,
# small enough to keep peak memory bounded for multi-GB capsule tarballs.
_SHA256_CHUNK = 1024 * 1024


@dataclass
class ManifestEntry:
    """One ``files:`` entry — a relative path + content hash + size."""

    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass
class Manifest:
    """A full ``MANIFEST.yaml`` document for one cohort."""

    id: str
    name: str
    version: str
    source_url: str
    benchmark: str
    mask_seed: str = ""
    files: list[ManifestEntry] = field(default_factory=list)
    scitex_dataset_version: str = ""
    prepared_at: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "source_url": self.source_url,
            "benchmark": self.benchmark,
            "mask_seed": self.mask_seed,
            "files": [e.as_dict() for e in self.files],
            "scitex_dataset_version": self.scitex_dataset_version,
            "prepared_at": self.prepared_at,
        }


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(_SHA256_CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def manifest_entries_for(paths: Iterable[Path], root: Path) -> list[ManifestEntry]:
    """Build sorted ``ManifestEntry`` rows for the given paths.

    ``path`` field is rendered relative to ``root`` with POSIX
    separators so the manifest is portable across hosts.
    """
    entries: list[ManifestEntry] = []
    for p in paths:
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        entries.append(
            ManifestEntry(path=rel, sha256=sha256_file(p), size=p.stat().st_size)
        )
    # sort by path so the YAML round-trips deterministically
    entries.sort(key=lambda e: e.path)
    return entries


def _scitex_dataset_version() -> str:
    """Resolve the running ``scitex-dataset`` version (empty on dev checkouts)."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _v

        try:
            return _v("scitex-dataset")
        except PackageNotFoundError:
            return ""
    except ImportError:  # pragma: no cover — only on ancient Pythons
        return ""


def _yaml_scalar(value) -> str:
    """Render a single YAML scalar — string-quoted only when ambiguous.

    Numbers stay bare; strings get double-quoted with ``"`` and ``\\``
    escaped. Empty strings render as ``""`` so PyYAML / ruamel round-
    trip the absent-ness correctly.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    s = str(value)
    if s == "" or any(ch in s for ch in " :#\"'\\\n\t"):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def to_yaml(doc: dict) -> str:
    """Render the manifest dict to YAML using only stdlib.

    Schema is fixed and shallow (top-level scalars + one list of dicts),
    so a tiny hand-rolled emitter is enough — no PyYAML dependency.
    The output is deterministic: scalars rendered in the order
    ``doc.items()`` provides, ``files:`` rows in their list order
    (callers pre-sort by path via ``manifest_entries_for``).
    """
    lines: list[str] = []
    for key, value in doc.items():
        if key == "files":
            lines.append("files:")
            if not value:
                # Keep the empty list explicit so YAML readers don't
                # collapse the key.
                lines[-1] = "files: []"
                continue
            for entry in value:
                # Each entry is a flat dict; emit as a "- " list item
                # with two trailing 2-space-indented scalars.
                items = list(entry.items())
                first_k, first_v = items[0]
                lines.append(f"  - {first_k}: {_yaml_scalar(first_v)}")
                for k2, v2 in items[1:]:
                    lines.append(f"    {k2}: {_yaml_scalar(v2)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    # Trailing newline so the file matches conventional POSIX text
    # behaviour and round-trips cleanly through ``read_text`` /
    # ``write_text``.
    return "\n".join(lines) + "\n"


def write_manifest(
    *,
    manifest_dir: Path,
    id: str,
    name: str,
    version: str,
    source_url: str,
    benchmark: str,
    tracked_paths: Iterable[Path],
    tracked_root: Path,
    mask_seed: str = "",
    prepared_at: str | None = None,
) -> Path:
    """Compute and write ``manifest_dir/MANIFEST.yaml``.

    ``tracked_paths`` are the artifacts whose presence + content
    identifies this snapshot — typically the masked questions file and
    any inventory JSON. They are sha256-hashed relative to
    ``tracked_root``. The returned ``Path`` is the manifest file
    written.

    Idempotent across re-runs IF ``prepared_at`` is pinned (callers
    pass a fixed string when reproducing a snapshot byte-for-byte);
    when ``prepared_at`` is None the current UTC time is recorded.
    """
    if prepared_at is None:
        prepared_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entries = manifest_entries_for(tracked_paths, tracked_root)
    doc = Manifest(
        id=id,
        name=name,
        version=version,
        source_url=source_url,
        benchmark=benchmark,
        mask_seed=mask_seed,
        files=entries,
        scitex_dataset_version=_scitex_dataset_version(),
        prepared_at=prepared_at,
    ).as_dict()

    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / MANIFEST_FILENAME
    out.write_text(to_yaml(doc), encoding="utf-8")
    # Answer-bearing upstream artifacts are never tracked here — by
    # construction ``tracked_paths`` comes from the agent-visible masked
    # dir, not the operator-private raw dir.
    _ = os  # silence unused-import linter if os ever gets dropped above
    return out


__all__ = [
    "MANIFEST_FILENAME",
    "ManifestEntry",
    "Manifest",
    "manifest_entries_for",
    "sha256_file",
    "to_yaml",
    "write_manifest",
]

# EOF
