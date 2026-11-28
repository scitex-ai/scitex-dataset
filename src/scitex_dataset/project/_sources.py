#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_sources.py

"""The PUBLIC datasets a project uses: ``<root>/.scitex/dataset/sources.json``.

One small JSON manifest travels with the project. Each entry pins the chosen
dataset — repository, accession, version, DOI, licence, citation — and its
file list with SHA-256 and size, plus where the project keeps a local copy
(``local_dir``, under ``data/``) and the state of its fetch::

    {"schema": "scitex-dataset.sources", "version": 1,
     "datasets": [{"key": "reference:mnist-idx", "repository": "reference",
                   "accession": "mnist-idx", "title": ..., "version": ...,
                   "doi": ..., "url": ..., "license": ..., "citation": ...,
                   "files": [{"path", "sha256", "size_bytes", "url"}],
                   "local_dir": "data/mnist" | null,
                   "fetch": {"state", "job_id", "mode", "fetched_at", "error"},
                   "origin": "app" | "detected" | "migrated:registry.json",
                   "added_by", "added_at"}]}

Besides what is RECORDED, :func:`detect_datasets` recognises pinned public
data already on disk: a file under ``data/`` whose SHA-256 equals a reference
file's is that reference dataset, whoever put it there. Only files whose size
matches a pinned size are hashed, so detection costs almost nothing.

The 0.15 ``registry.json`` held arbitrary project paths and URLs. Its URL
registrations are carried into this manifest (``origin:
"migrated:registry.json"``) — shown on read, written on the next write; path
registrations are project files, which belong to the Hub's file browser.

Writes are serialized with an exclusive ``flock`` and land with
``os.replace``; reads never create anything.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ._paths import DATA_DIR, DatasetError, inside, project_root, sha256_of, slug

SOURCES_RELPATH = ".scitex/dataset/sources.json"
SOURCES_SCHEMA = "scitex-dataset.sources"
SOURCES_VERSION = 1

#: Bound on the files walked under ``data/`` for detection.
MAX_SCANNED_FILES = 20_000

_ENTRY_FIELDS = (
    "key",
    "repository",
    "accession",
    "title",
    "version",
    "doi",
    "url",
    "license",
    "citation",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty() -> dict:
    return {"schema": SOURCES_SCHEMA, "version": SOURCES_VERSION, "datasets": []}


def sources_path(root: Path) -> Path:
    root = Path(root).resolve()
    path = (root / SOURCES_RELPATH).resolve()
    if not path.is_relative_to(root):
        raise DatasetError("the dataset manifest resolves outside the project", 403)
    return path


def _read(root: Path) -> dict:
    path = sources_path(root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _empty()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise DatasetError(f"{SOURCES_RELPATH} is not valid JSON: {exc}", 409) from exc
    if not isinstance(data, dict) or data.get("schema") != SOURCES_SCHEMA:
        raise DatasetError(f"{SOURCES_RELPATH} is not a dataset manifest", 409)
    if data.get("version") != SOURCES_VERSION:
        raise DatasetError(
            f"{SOURCES_RELPATH} has version {data.get('version')!r}; "
            f"this scitex-dataset reads version {SOURCES_VERSION}",
            409,
        )
    if not isinstance(data.get("datasets"), list):
        raise DatasetError(f"{SOURCES_RELPATH} has no 'datasets' list", 409)
    return data


# --------------------------------------------------------------------------
# migration from the 0.15 registry
# --------------------------------------------------------------------------


def _url_key(url: str) -> str:
    return f"url:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}"


def pending_migrations(root: Path, manifest: Optional[dict] = None) -> List[dict]:
    """URL registrations of a legacy ``registry.json`` not yet in the manifest."""
    from ._registry import RegistryError, load_registry

    try:
        registry = load_registry(root)
    except RegistryError:
        return []
    manifest = manifest if manifest is not None else _read(root)
    known = {d.get("key") for d in manifest["datasets"]}
    out = []
    for item in registry.get("datasets", []):
        if item.get("kind") != "url" or not item.get("location"):
            continue
        key = _url_key(item["location"])
        if key in known:
            continue
        out.append(
            {
                "key": key,
                "repository": "url",
                "accession": item["location"],
                "title": item.get("name") or item["location"],
                "version": None,
                "doi": None,
                "url": item["location"],
                "license": None,
                "citation": item.get("note") or None,
                "files": [],
                "local_dir": None,
                "fetch": {"state": "not-fetched"},
                "origin": "migrated:registry.json",
                "added_by": item.get("registered_by"),
                "added_at": item.get("registered_at"),
            }
        )
    return out


def load_sources(root) -> dict:
    """The manifest, with pending legacy URL registrations merged in. Never writes."""
    root = project_root(root)
    manifest = _read(root)
    manifest["datasets"] = list(manifest["datasets"]) + pending_migrations(
        root, manifest
    )
    return manifest


@contextmanager
def locked_sources(root) -> Iterator[dict]:
    """Read-modify-write the manifest under an exclusive lock.

    Pending legacy URL registrations are migrated into the document first, so
    the first write of a 0.15 project carries them over.
    """
    root = project_root(root)
    path = sources_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            data = _read(root)
            data["datasets"].extend(pending_migrations(root, data))
            yield data
            fd, tmp = tempfile.mkstemp(
                dir=path.parent, prefix=".sources-", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _find(data: dict, key: str) -> Optional[dict]:
    for entry in data["datasets"]:
        if entry.get("key") == key:
            return entry
    return None


def default_local_dir(card: Dict[str, Any]) -> str:
    """``data/<slug>`` for a dataset that has none yet."""
    return f"{DATA_DIR}/{slug(card.get('accession') or card['key'])}"


def record_dataset(
    root,
    card: Dict[str, Any],
    *,
    local_dir: Optional[str] = None,
    added_by: Optional[str] = None,
    origin: str = "app",
) -> dict:
    """Record ``card`` as used by the project; idempotent per dataset key.

    ``local_dir`` must be under ``data/``; it defaults to ``data/<accession>``.
    Re-recording a dataset refreshes its metadata and keeps its fetch state.
    """
    root = project_root(root)
    local_dir = (local_dir or default_local_dir(card)).strip("/")
    target = inside(root, local_dir)
    if not target.is_relative_to(root / DATA_DIR):
        raise DatasetError(f"a dataset's local folder must be under {DATA_DIR}/", 400)
    files = [
        {
            "path": f["path"],
            "sha256": f.get("sha256"),
            "size_bytes": f.get("size_bytes"),
            "url": f.get("url"),
        }
        for f in card.get("files") or []
    ]
    with locked_sources(root) as data:
        entry = _find(data, card["key"])
        if entry is None:
            entry = {
                "fetch": {"state": "not-fetched"},
                "origin": origin,
                "added_by": added_by,
                "added_at": _now(),
            }
            data["datasets"].append(entry)
        for field in _ENTRY_FIELDS:
            entry[field] = card.get(field)
        if files or not entry.get("files"):
            entry["files"] = files
        entry["local_dir"] = target.relative_to(root).as_posix()
        entry["updated_at"] = _now()
        return dict(entry)


def update_fetch(root, key: str, **fields) -> dict:
    """Merge ``fields`` into the ``fetch`` block of dataset ``key``."""
    files = fields.pop("files", None)
    with locked_sources(root) as data:
        entry = _find(data, key)
        if entry is None:
            raise DatasetError("this dataset is not recorded in the project", 404)
        entry.setdefault("fetch", {}).update(fields)
        if files is not None:
            entry["files"] = files
        return dict(entry)


# --------------------------------------------------------------------------
# detection of pinned public data already on disk
# --------------------------------------------------------------------------


def _provenance_json(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``{filename: {url, md5}}`` from a download-provenance file, if present."""
    found: Dict[str, Dict[str, Any]] = {}
    for name in ("download-provenance.json", "provenance.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rows = data.get("files") if isinstance(data, dict) else data
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            filename = row.get("file") or row.get("path") or row.get("filename")
            if filename:
                found[Path(str(filename)).name] = {
                    "url": row.get("source_url") or row.get("url"),
                    "provenance_file": path.name,
                }
    return found


def detect_datasets(root) -> List[dict]:
    """Reference datasets whose pinned bytes are already under ``data/``.

    Returns one entry per (reference, directory) with the matched files, their
    project-relative paths and the provenance file found next to them.
    """
    from ..general.reference import REFERENCES, reference_files

    root = project_root(root)
    data_dir = root / DATA_DIR
    if not data_dir.is_dir() or not data_dir.resolve().is_relative_to(root):
        return []
    pinned: Dict[int, List[tuple]] = {}
    for ref_id, entry in REFERENCES.items():
        for item in reference_files(entry):
            if isinstance(item.get("size_bytes"), int):
                pinned.setdefault(item["size_bytes"], []).append((ref_id, item))

    hits: Dict[tuple, Dict[str, Path]] = {}
    scanned = 0
    for current, dirs, files in os.walk(data_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            scanned += 1
            if scanned > MAX_SCANNED_FILES:
                break
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            candidates = pinned.get(path.stat().st_size)
            if not candidates:
                continue
            digest = sha256_of(path)
            for ref_id, item in candidates:
                if item["sha256"] == digest:
                    hits.setdefault((ref_id, Path(current)), {})[item["path"]] = path

    detected = []
    for (ref_id, directory), matched in sorted(hits.items()):
        wanted = [f["path"] for f in reference_files(REFERENCES[ref_id])]
        if set(matched) != set(wanted):
            continue  # a partial set is not this dataset
        provenance = _provenance_json(directory)
        detected.append(
            {
                "key": f"reference:{ref_id}",
                "local_dir": directory.relative_to(root).as_posix(),
                "paths": [matched[p].relative_to(root).as_posix() for p in wanted],
                "provenance_file": (
                    (directory / next(iter(provenance.values()))["provenance_file"])
                    .relative_to(root)
                    .as_posix()
                    if provenance
                    else None
                ),
            }
        )
    return detected


# --------------------------------------------------------------------------
# local state of a recorded dataset
# --------------------------------------------------------------------------


def local_state(root: Path, entry: dict) -> Dict[str, Any]:
    """Compare the dataset's pinned files with what is in its ``local_dir``.

    ``state``: ``verified`` (every file present with its SHA-256),
    ``changed`` (a file differs), ``missing`` (some or all absent) or
    ``not-local`` (no folder / no file list).
    """
    local_dir = entry.get("local_dir")
    files = entry.get("files") or []
    if not local_dir or not files:
        return {"state": "not-local", "files": []}
    from ..catalog import human_size

    base = inside(root, local_dir)
    rows = []
    for item in files:
        path = base / item["path"]
        row = {
            **item,
            "local_path": path.relative_to(root).as_posix(),
            "size": human_size(item.get("size_bytes")),
        }
        if not path.is_file():
            row["state"] = "missing"
        elif item.get("sha256") and sha256_of(path) != item["sha256"]:
            row["state"] = "changed"
        else:
            row["state"] = "verified"
        rows.append(row)
    states = {r["state"] for r in rows}
    if states == {"verified"}:
        state = "verified"
    elif "changed" in states:
        state = "changed"
    else:
        state = "missing"
    return {"state": state, "files": rows}


__all__ = [
    "SOURCES_RELPATH",
    "SOURCES_SCHEMA",
    "default_local_dir",
    "detect_datasets",
    "load_sources",
    "local_state",
    "locked_sources",
    "pending_migrations",
    "record_dataset",
    "update_fetch",
]

# EOF
