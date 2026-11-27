#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_paths.py

"""Containment and checksums for everything this package writes into a project.

* a path is resolved (symlinks followed) and must stay inside the project
  root; a ``data/`` symlink to elsewhere is refused, not followed;
* checksums are SHA-256, cached per (path, size, mtime) for the life of the
  process, so listing a project twice does not re-read unchanged bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path

#: Where a scitex-template project keeps its data.
DATA_DIR = "data"

_CHUNK = 1024 * 1024
_HASH_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_HASH_CACHE_MAX = 8192
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetError(ValueError):
    """A refused dataset operation, with the HTTP-like status a host should use."""

    def __init__(self, message: str, status: int = 400):
        self.status = status
        super().__init__(message)


def project_root(root) -> Path:
    """The resolved project root; 404 when it is not a directory."""
    path = Path(root).resolve()
    if not path.is_dir():
        raise DatasetError("project workspace not found", 404)
    return path


def inside(root: Path, relative: str) -> Path:
    """``root / relative`` resolved, refused when it escapes ``root``."""
    if (
        not isinstance(relative, str)
        or not relative
        or "\x00" in relative
        or "\\" in relative
    ):
        raise DatasetError("invalid path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise DatasetError("path is outside the project", 403)
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetError("path is outside the project", 403)
    return resolved


def slug(text: str) -> str:
    """A safe single path component derived from ``text``."""
    value = _SLUG_RE.sub("-", (text or "").split("/")[-1].split(":")[-1]).strip("-.")
    return value[:80] or "dataset"


def sha256_of(path: Path) -> str:
    """SHA-256 of ``path``, cached by (path, size, mtime)."""
    stat = path.stat()
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


def same_device(a: Path, b: Path) -> bool:
    """Whether ``a`` and ``b`` (existing paths) are on one filesystem."""
    return os.stat(a).st_dev == os.stat(b).st_dev


# EOF
