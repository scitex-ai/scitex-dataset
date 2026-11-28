#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/cache/__init__.py

"""The LOCAL DATASET CACHE: a public dataset fetched once, reused everywhere.

One directory per (repository, accession, version), shared by every project
and user of the host::

    <cache-root>/<repository>/<accession>/<version>/
        files/          the dataset's bytes, exactly as fetched
        entry.json      what is here: files with SHA-256 and size, state,
                        licence, citation, fetched_at, last_used_at
        .lock           held while a fetch writes into ``files/``

``<cache-root>`` is ``$SCITEX_DATASET_CACHE`` when set (a host points it at a
shared volume), else ``<user-scope>/cache`` (``~/.scitex/dataset/cache``).

States
    ``cached``      every expected file is present with its recorded size;
    ``partial``     a fetch was interrupted or failed part-way;
    ``not-cached``  nothing is here.

Eviction is least-recently-used down to ``cache.max_bytes`` of the dataset
config (:mod:`scitex_dataset._config`). An entry being fetched (its lock is
held) is never evicted. A project that USES a dataset holds hard links to the
cached bytes (see :mod:`scitex_dataset.project`), so evicting the cache entry
never breaks the project's copy.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..catalog import human_size
from ..project._paths import sha256_of, slug

__all__ = [
    "CACHE_ENV",
    "CacheBusy",
    "DatasetCache",
    "cache_root",
    "eviction_policy",
]

CACHE_ENV = "SCITEX_DATASET_CACHE"
ENTRY_SCHEMA = "scitex-dataset.cache-entry"
ENTRY_VERSION = 1

#: ``progress(done_bytes, total_bytes_or_None, message)``.
Progress = Callable[[int, Optional[int], str], None]


class CacheBusy(RuntimeError):
    """Another process is fetching the same cache entry right now."""


def cache_root() -> Path:
    """Where the shared cache lives on this host."""
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    from .._config import user_root

    return user_root() / "cache"


def eviction_policy() -> Dict[str, Any]:
    """``{"policy": "lru", "max_bytes": int | None}`` from the dataset config."""
    from .._config import load_config

    section = (load_config() or {}).get("cache") or {}
    max_bytes = section.get("max_bytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        max_bytes = None
    return {"policy": "lru", "max_bytes": max_bytes, "max": human_size(max_bytes)}


def _now() -> str:
    # Microseconds: the LRU order must not tie within a second.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".entry-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class DatasetCache:
    """The cache at ``root`` (default :func:`cache_root`)."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else cache_root()

    # ------------------------------------------------------------------ layout
    def entry_dir(self, key: str, version: Optional[str]) -> Path:
        repository, _, accession = key.partition(":")
        return (
            self.root / slug(repository) / slug(accession) / slug(version or "latest")
        )

    def files_dir(self, key: str, version: Optional[str]) -> Path:
        return self.entry_dir(key, version) / "files"

    def _entry_path(self, key: str, version: Optional[str]) -> Path:
        return self.entry_dir(key, version) / "entry.json"

    def read_entry(self, key: str, version: Optional[str]) -> Optional[dict]:
        path = self._entry_path(key, version)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None
        return data if data.get("schema") == ENTRY_SCHEMA else None

    # ------------------------------------------------------------------ status
    def status(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Cache state of the dataset ``card`` (its key and version)."""
        key, version = card["key"], card.get("version")
        entry = self.read_entry(key, version)
        if entry is None:
            return {"state": "not-cached", "version": version, "size": None}
        files_dir = self.files_dir(key, version)
        present = 0
        on_disk = 0
        for item in entry.get("files", []):
            path = files_dir / item["path"]
            if path.is_file() and path.stat().st_size == item.get("size_bytes"):
                present += 1
                on_disk += item["size_bytes"]
        expected = len(entry.get("files", []))
        complete = entry.get("state") == "complete" and expected and present == expected
        return {
            "state": "cached" if complete else "partial",
            "version": entry.get("dataset_version"),
            "files_present": present,
            "files_expected": expected,
            "size_bytes": on_disk,
            "size": human_size(on_disk),
            "fetched_at": entry.get("fetched_at"),
            "last_used_at": entry.get("last_used_at"),
            "path": str(files_dir),
        }

    def entries(self) -> List[Dict[str, Any]]:
        """Every entry with its on-disk size, most recently used first."""
        rows = []
        if not self.root.is_dir():
            return rows
        for entry_json in self.root.glob("*/*/*/entry.json"):
            try:
                data = json.loads(entry_json.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if data.get("schema") != ENTRY_SCHEMA:
                continue
            size = sum(
                p.stat().st_size
                for p in (entry_json.parent / "files").rglob("*")
                if p.is_file()
            )
            rows.append(
                {
                    "key": data.get("key"),
                    "title": data.get("title"),
                    "version": data.get("dataset_version"),
                    "state": data.get("state"),
                    "size_bytes": size,
                    "size": human_size(size),
                    "fetched_at": data.get("fetched_at"),
                    "last_used_at": data.get("last_used_at") or data.get("fetched_at"),
                    "busy": self._is_locked(entry_json.parent),
                    "dir": str(entry_json.parent),
                }
            )
        rows.sort(key=lambda r: r["last_used_at"] or "", reverse=True)
        return rows

    def usage(self) -> Dict[str, Any]:
        rows = self.entries()
        total = sum(r["size_bytes"] for r in rows)
        return {
            "root": str(self.root),
            "entries": len(rows),
            "size_bytes": total,
            "size": human_size(total),
            **eviction_policy(),
        }

    # ------------------------------------------------------------------- locks
    @staticmethod
    def _is_locked(entry_dir: Path) -> bool:
        lock_path = entry_dir / ".lock"
        if not lock_path.exists():
            return False
        with open(lock_path, "a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return False

    @contextmanager
    def _locked(self, entry_dir: Path) -> Iterator[None]:
        entry_dir.mkdir(parents=True, exist_ok=True)
        with open(entry_dir / ".lock", "a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CacheBusy(
                    "this dataset is being fetched by another job; try again when it finishes"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------- fetch
    def touch(self, key: str, version: Optional[str]) -> None:
        """Mark an entry as just used (the LRU clock)."""
        path = self._entry_path(key, version)
        entry = self.read_entry(key, version)
        if entry is not None:
            entry["last_used_at"] = _now()
            _write_json(path, entry)

    def fetch(
        self, card: Dict[str, Any], progress: Optional[Progress] = None
    ) -> Dict[str, Any]:
        """Bring ``card``'s dataset into the cache; reuse what is already there.

        Returns the entry document. Files are verified by SHA-256 when the
        card pins one (reference files); otherwise their SHA-256 is computed
        and recorded, so every cached byte has a checksum.
        """
        key, version = card["key"], card.get("version")
        entry_dir = self.entry_dir(key, version)
        report = progress or (lambda done, total, message: None)
        with self._locked(entry_dir):
            if self.status(card)["state"] == "cached":
                self.touch(key, version)
                report(1, 1, "already in the shared cache")
                return self.read_entry(key, version)
            files_dir = entry_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "schema": ENTRY_SCHEMA,
                "version": ENTRY_VERSION,
                "key": key,
                "title": card.get("title"),
                "repository": card.get("repository"),
                "accession": card.get("accession"),
                "dataset_version": version,
                "doi": card.get("doi"),
                "url": card.get("url"),
                "license": card.get("license"),
                "citation": card.get("citation"),
                "state": "partial",
                "files": [],
                "fetched_at": None,
                "last_used_at": _now(),
            }
            _write_json(entry_dir / "entry.json", entry)
            entry["files"] = _download(card, files_dir, report)
            entry["state"] = "complete"
            entry["fetched_at"] = entry["last_used_at"] = _now()
            _write_json(entry_dir / "entry.json", entry)
            return entry

    # ----------------------------------------------------------------- evict
    def evict(
        self, max_bytes: Optional[int] = None, *, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Remove least-recently-used entries until the cache fits ``max_bytes``.

        ``max_bytes`` defaults to the configured limit; with neither, nothing
        is evicted. Entries being fetched are skipped.
        """
        limit = max_bytes if max_bytes is not None else eviction_policy()["max_bytes"]
        rows = self.entries()
        total = sum(r["size_bytes"] for r in rows)
        evicted: List[Dict[str, Any]] = []
        if limit is None:
            return {"limit_bytes": None, "evicted": evicted, "size_bytes": total}
        for row in sorted(rows, key=lambda r: r["last_used_at"] or ""):
            if total <= limit:
                break
            if row["busy"]:
                continue
            if not dry_run:
                shutil.rmtree(row["dir"])
            total -= row["size_bytes"]
            evicted.append(
                {k: row[k] for k in ("key", "version", "size_bytes", "size")}
            )
        return {
            "limit_bytes": limit,
            "limit": human_size(limit),
            "evicted": evicted,
            "size_bytes": total,
            "size": human_size(total),
            "dry_run": dry_run,
        }


def _download(card: Dict[str, Any], files_dir: Path, report: Progress) -> List[dict]:
    """Fetch the dataset's files into ``files_dir``; return them with checksums."""
    repository, accession = card["repository"], card["accession"]
    if repository == "reference" or (
        card.get("files")
        and all(f.get("url") and f.get("sha256") for f in card["files"])
    ):
        return _download_pinned(card["files"], files_dir, card, report)
    if repository in ("gin", "huggingface"):
        from .. import download_dataset

        report(0, None, f"downloading {accession} from {card.get('repository_name')}")
        download_dataset(repository, accession, files_dir)
        return _checksum_tree(files_dir, report)
    raise NotImplementedError(
        f"{card.get('repository_name') or repository} is searchable but scitex-dataset "
        "cannot download its files yet"
    )


def _download_pinned(
    files: List[dict], files_dir: Path, card: Dict[str, Any], report: Progress
) -> List[dict]:
    from .._fetch_file import fetch_file

    total = sum(int(f.get("size_bytes") or 0) for f in files) or None
    done_before = 0
    out = []
    for item in files:
        target = (files_dir / item["path"]).resolve()
        if not target.is_relative_to(files_dir.resolve()):
            raise ValueError(f"file path escapes the cache entry: {item['path']!r}")

        def chunk(done, _size, _name=item["path"], _base=done_before):
            report(_base + done, total, f"downloading {_name}")

        record = fetch_file(
            item["url"],
            item["sha256"],
            target.parent,
            target.name,
            license=card.get("license"),
            citation=card.get("citation"),
            progress=chunk,
        )
        done_before += record.size_bytes
        report(done_before, total, f"verified {item['path']}")
        out.append(
            {
                "path": item["path"],
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "url": item["url"],
            }
        )
    return out


def _checksum_tree(files_dir: Path, report: Progress) -> List[dict]:
    paths = sorted(p for p in files_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in paths) or None
    done = 0
    out = []
    for path in paths:
        size = path.stat().st_size
        out.append(
            {
                "path": path.relative_to(files_dir).as_posix(),
                "sha256": sha256_of(path),
                "size_bytes": size,
                "url": None,
            }
        )
        done += size
        report(done, total, f"checksummed {path.name}")
    return out


# EOF
