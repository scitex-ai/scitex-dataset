#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_inventory.py

"""List, register and upload a project's datasets — given an authorized root.

Containment rules, applied to every path (read or write):

* a path is resolved (symlinks followed) and must stay inside the project
  root; a ``data/`` symlink to elsewhere is refused, not followed;
* names coming from a caller are single path components from a strict
  pattern — no separators, no leading dot, no ``..``;
* an upload never overwrites: the bytes go to a temporary file in the target
  directory and are published with ``os.link``, which fails if the name
  exists, so a reader never sees a partial dataset.

Checksums are SHA-256. They are cached per (path, size, mtime) for the life of
the process, so listing a project twice does not re-read unchanged bytes.
Files larger than :data:`MAX_HASH_BYTES` are listed with their size and a
``sha256`` of ``None`` plus a reason, instead of stalling a page load.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Optional
from urllib.parse import urlparse

from ._registry import RegistryError, load_registry, locked_registry

#: Where a scitex-template project keeps its data (``data/raw``, ``data/processed``…).
DATA_DIR = "data"
#: Files above this are listed without a checksum (reason given) on a page load.
MAX_HASH_BYTES = 512 * 1024 * 1024
#: Inventory bound: a listing reports ``truncated`` instead of walking forever.
MAX_LISTED_FILES = 2000
#: One upload's size cap; larger datasets are registered by path or URL instead.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
#: Free-text limits for a registration.
MAX_NAME_CHARS = 120
MAX_NOTE_CHARS = 1000

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHUNK = 1024 * 1024
_HASH_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_HASH_CACHE_MAX = 8192


class DatasetError(ValueError):
    """A refused dataset operation, with the HTTP-like status a host should use."""

    def __init__(self, message: str, status: int = 400):
        self.status = status
        super().__init__(message)


# ---------------------------------------------------------------------------
# paths and checksums
# ---------------------------------------------------------------------------


def _root(root) -> Path:
    path = Path(root).resolve()
    if not path.is_dir():
        raise DatasetError("project workspace not found", 404)
    return path


def _inside(root: Path, relative: str) -> Path:
    """``root / relative`` resolved, refused when it escapes ``root``."""
    if not isinstance(relative, str) or "\x00" in relative or "\\" in relative:
        raise DatasetError("invalid path")
    candidate = Path(relative)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise DatasetError("path is outside the project", 403) from exc
    if any(part == ".." for part in candidate.parts):
        raise DatasetError("path is outside the project", 403)
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetError("path is outside the project", 403)
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path, stat: os.stat_result) -> Optional[str]:
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        _HASH_CACHE.move_to_end(key)
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    if len(_HASH_CACHE) > _HASH_CACHE_MAX:
        _HASH_CACHE.popitem(last=False)
    return value


def _file_facts(path: Path) -> dict:
    stat = path.stat()
    facts = {"size": stat.st_size, "sha256": None, "sha256_note": None}
    if stat.st_size > MAX_HASH_BYTES:
        facts["sha256_note"] = (
            f"not hashed on request: larger than {MAX_HASH_BYTES // (1024 * 1024)} MiB"
        )
    else:
        facts["sha256"] = _sha256(path, stat)
    return facts


def _walk_files(root: Path, top: Path, limit: int) -> tuple[list[Path], bool]:
    """Regular files under ``top`` that stay inside ``root``; hidden entries skipped."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            if len(found) >= limit:
                return found, True
            found.append(path)
    return found, False


def _tree_facts(root: Path, top: Path) -> dict:
    """Size and a tree digest (sha256 over ``relpath\\0sha256\\n`` lines) of a directory."""
    files, truncated = _walk_files(root, top, MAX_LISTED_FILES)
    total = 0
    lines = []
    unhashed = False
    for path in files:
        facts = _file_facts(path)
        total += facts["size"]
        if facts["sha256"] is None:
            unhashed = True
        lines.append(f"{path.relative_to(top).as_posix()}\0{facts['sha256']}\n")
    out = {"size": total, "files": len(files), "sha256": None, "sha256_note": None}
    if truncated:
        out["sha256_note"] = f"not hashed: more than {MAX_LISTED_FILES} files"
    elif unhashed:
        out["sha256_note"] = "not hashed: contains a file too large to hash on request"
    else:
        out["sha256"] = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return out


def _facts(root: Path, path: Path) -> dict:
    return _tree_facts(root, path) if path.is_dir() else _file_facts(path)


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------


def _registered_view(root: Path, entry: dict) -> dict:
    view = {
        key: entry.get(key)
        for key in (
            "id",
            "name",
            "kind",
            "location",
            "note",
            "registered_at",
            "registered_by",
        )
    }
    view["recorded_sha256"] = entry.get("sha256")
    view["size"] = entry.get("size")
    view["sha256"] = None
    view["sha256_note"] = None
    if entry.get("kind") == "url":
        view["state"] = "external"
        return view
    try:
        path = _inside(root, entry.get("location") or "")
    except DatasetError:
        view["state"] = "outside-project"
        return view
    if not path.exists():
        view["state"] = "missing"
        return view
    facts = _facts(root, path)
    view.update(
        size=facts["size"], sha256=facts["sha256"], sha256_note=facts["sha256_note"]
    )
    if facts["sha256"] is None or entry.get("sha256") is None:
        view["state"] = "unverified"
    elif facts["sha256"] == entry.get("sha256"):
        view["state"] = "unchanged"
    else:
        view["state"] = "changed"
    return view


def list_datasets(root, *, provenance: bool = True) -> dict:
    """The project's datasets: files under ``data/`` plus registered datasets.

    Returns ``{"data_dir", "files", "truncated", "registered", "registry_error",
    "provenance"}``. Paths are project-relative POSIX strings; the absolute
    project root is never part of the result.
    """
    root = _root(root)
    try:
        registry = load_registry(root)
        registry_error = None
    except RegistryError as exc:
        registry, registry_error = {"datasets": []}, str(exc)
    registered_paths = {
        entry.get("location")
        for entry in registry["datasets"]
        if entry.get("kind") == "path"
    }

    files: list[dict] = []
    truncated = False
    data_dir = root / DATA_DIR
    if data_dir.is_dir() and data_dir.resolve().is_relative_to(root):
        paths, truncated = _walk_files(root, data_dir, MAX_LISTED_FILES)
        for path in paths:
            relative = _relative(root, path)
            files.append(
                {
                    "path": relative,
                    **_file_facts(path),
                    "registered": relative in registered_paths,
                }
            )

    registered = [_registered_view(root, entry) for entry in registry["datasets"]]

    result = {
        "data_dir": DATA_DIR,
        "files": files,
        "truncated": truncated,
        "registered": registered,
        "registry_error": registry_error,
        "provenance": None,
    }
    if provenance:
        from .provenance import readers_of

        local = [f["path"] for f in files]
        local += [
            r["location"]
            for r in registered
            if r["kind"] == "path" and r["state"] not in ("missing", "outside-project")
        ]
        result["provenance"] = readers_of(root, sorted(set(local)))
    return result


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value, limit: int, what: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DatasetError(f"{what} must be text")
    value = value.strip()
    if len(value) > limit:
        raise DatasetError(f"{what} is longer than {limit} characters")
    return value


def _dataset_id(registry: dict, name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "dataset"
    taken = {entry.get("id") for entry in registry["datasets"]}
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def register_dataset(
    root,
    location: str,
    *,
    kind: Optional[str] = None,
    name: str = "",
    note: str = "",
    registered_by: Optional[str] = None,
) -> dict:
    """Register a path already in the project, or an external http(s) URL.

    A path must exist; its size and SHA-256 are recorded now, so the listing
    can later say whether the bytes changed. A URL is recorded as given and
    never fetched — registering is not downloading. ``kind`` (``"path"`` or
    ``"url"``) pins what ``location`` must be; ``None`` infers it.
    """
    root = _root(root)
    location = _clean_text(location, 2048, "location")
    if not location:
        raise DatasetError("a path or URL is required")
    name = _clean_text(name, MAX_NAME_CHARS, "name")
    note = _clean_text(note, MAX_NOTE_CHARS, "note")
    if kind not in (None, "path", "url"):
        raise DatasetError("kind must be 'path' or 'url'")

    parsed = urlparse(location)
    is_url = bool(parsed.scheme) and len(parsed.scheme) > 1
    if kind is not None and (kind == "url") != is_url:
        raise DatasetError(
            "expected an http(s) URL"
            if kind == "url"
            else "expected a project path, not a URL"
        )
    if is_url:
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            raise DatasetError("only http(s) URLs or project paths can be registered")
        entry = {"kind": "url", "location": location, "size": None, "sha256": None}
        default_name = Path(parsed.path).name or parsed.netloc
    else:
        path = _inside(root, location)
        if not path.exists():
            raise DatasetError(f"{location!r} does not exist in this project", 404)
        if path == root:
            raise DatasetError("register a file or a directory inside the project")
        facts = _facts(root, path)
        entry = {
            "kind": "path",
            "location": _relative(root, path),
            "size": facts["size"],
            "sha256": facts["sha256"],
        }
        default_name = path.name

    with locked_registry(root) as registry:
        for existing in registry["datasets"]:
            if (
                existing.get("kind") == entry["kind"]
                and existing.get("location") == entry["location"]
            ):
                raise DatasetError(f"{entry['location']!r} is already registered", 409)
        entry.update(
            id=_dataset_id(registry, name or default_name),
            name=name or default_name,
            note=note,
            registered_at=_now(),
            registered_by=registered_by,
        )
        registry["datasets"].append(entry)
    return entry


def _safe_name(filename: str) -> str:
    name = Path(str(filename or "")).name
    if not _SAFE_COMPONENT.fullmatch(name) or name in (".", ".."):
        raise DatasetError(
            "file name must start with a letter or digit and use only "
            "letters, digits, '.', '_' or '-' (max 128 characters)"
        )
    return name


def save_upload(
    root,
    filename: str,
    chunks: Iterable[bytes] | BinaryIO,
    *,
    subdir: str = "",
) -> dict:
    """Write an uploaded file to ``data/[<subdir>/]<filename>``, never overwriting.

    ``chunks`` is an iterable of ``bytes`` (e.g. Django's
    ``UploadedFile.chunks()``) or a binary file object. Returns the new
    file's ``{"path", "size", "sha256"}``.
    """
    root = _root(root)
    name = _safe_name(filename)
    parts = [p for p in (subdir or "").split("/") if p]
    for part in parts:
        _safe_name(part)
    target_dir = _inside(root, "/".join([DATA_DIR, *parts]))
    target_dir.mkdir(parents=True, exist_ok=True)
    # mkdir may have followed a symlink planted between the check and here.
    if not target_dir.resolve().is_relative_to(root):
        raise DatasetError("path is outside the project", 403)
    destination = target_dir / name
    if destination.exists() or destination.is_symlink():
        raise DatasetError(f"{_relative(root, destination)!r} already exists", 409)

    if hasattr(chunks, "read"):
        reader = chunks
        chunks = iter(lambda: reader.read(_CHUNK), b"")
    digest = hashlib.sha256()
    size = 0
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".upload-")
    try:
        with os.fdopen(fd, "wb") as handle:
            for chunk in chunks:
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise DatasetError(
                        f"upload is larger than {MAX_UPLOAD_BYTES // (1024**3)} GiB; "
                        "register it by path or URL instead",
                        413,
                    )
                digest.update(chunk)
                handle.write(chunk)
        try:
            os.link(tmp, destination)
        except FileExistsError as exc:
            raise DatasetError(
                f"{_relative(root, destination)!r} already exists", 409
            ) from exc
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {
        "path": _relative(root, destination),
        "size": size,
        "sha256": digest.hexdigest(),
    }


# EOF
