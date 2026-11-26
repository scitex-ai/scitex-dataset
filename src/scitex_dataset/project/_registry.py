#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_registry.py

"""The project-local registry of explicitly registered datasets.

One JSON document at ``<root>/.scitex/dataset/registry.json`` (the project
scope of the SciTeX local-state-directories standard, next to the dataset
config). It travels with the project, so a registration made in the Hub is
visible to a script run in the same tree and vice versa.

Writes are serialized with an exclusive ``flock`` on a sibling lock file and
land with ``os.replace``, so two concurrent registrations can never lose one
another and a reader never sees a half-written document.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REGISTRY_RELPATH = ".scitex/dataset/registry.json"
REGISTRY_SCHEMA = "scitex-dataset.registry"
REGISTRY_VERSION = 1


class RegistryError(ValueError):
    """The registry document cannot be read or is not ours."""


def _empty() -> dict:
    return {"schema": REGISTRY_SCHEMA, "version": REGISTRY_VERSION, "datasets": []}


def registry_path(root: Path) -> Path:
    """The registry file, refused when it would resolve outside ``root``."""
    root = Path(root).resolve()
    path = (root / REGISTRY_RELPATH).resolve()
    if not path.is_relative_to(root):
        raise RegistryError("the dataset registry resolves outside the project")
    return path


def load_registry(root: Path) -> dict:
    """The registry document; an empty one when the project has none yet.

    Never creates anything: a read-only member listing datasets must not
    leave a file behind.
    """
    path = registry_path(root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _empty()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RegistryError(f"{REGISTRY_RELPATH} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"{REGISTRY_RELPATH} is not a dataset registry")
    if data.get("version") != REGISTRY_VERSION:
        raise RegistryError(
            f"{REGISTRY_RELPATH} has version {data.get('version')!r}; "
            f"this scitex-dataset reads version {REGISTRY_VERSION}"
        )
    if not isinstance(data.get("datasets"), list):
        raise RegistryError(f"{REGISTRY_RELPATH} has no 'datasets' list")
    return data


@contextmanager
def locked_registry(root: Path) -> Iterator[dict]:
    """Read-modify-write the registry under an exclusive lock.

    The caller mutates the yielded document; it is written back atomically
    when the block exits without an exception.
    """
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            data = load_registry(root)
            yield data
            fd, tmp = tempfile.mkstemp(
                dir=path.parent, prefix=".registry-", suffix=".json"
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


# EOF
