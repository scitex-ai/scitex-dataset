#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_config.py

"""Config-file resolution following the SciTeX local-state-directories standard.

Precedence (highest first):

1. CLI flag — caller passes ``config_path`` directly
2. ``$SCITEX_DATASET_CONFIG`` env var
3. Project scope — ``<project>/.scitex/dataset/config.yaml``
4. User scope — ``$SCITEX_DIR/dataset/config.yaml`` (default ``~/.scitex/dataset/config.yaml``)

Setting ``SCITEX_DIR`` relocates the user scope atomically — see the
``general/01_ecosystem_06_local-state-directories`` skill.

The runtime root is ``<scope-root>/runtime/`` and is where caches, logs,
PIDs, and the local SQLite index live.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

PKG_SHORT = "dataset"
ENV_CONFIG = "SCITEX_DATASET_CONFIG"
ENV_DIR = "SCITEX_DIR"


def user_root() -> Path:
    """User-scope root: ``$SCITEX_DIR/<pkg-short>`` (default ``~/.scitex/dataset``)."""
    base = Path(os.environ.get(ENV_DIR, str(Path.home() / ".scitex")))
    return base / PKG_SHORT


def project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` looking for ``<dir>/.scitex/dataset/``.

    Returns the first match (project scope), or ``None`` if not found.
    Stops at the filesystem root or when a ``.git`` directory is hit
    without a corresponding ``.scitex/dataset/`` — that's a project that
    hasn't opted into project-scope state, and we fall back to user
    scope.
    """
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / ".scitex" / PKG_SHORT
        if candidate.is_dir():
            return candidate
        if (d / ".git").exists() and d != here:
            # Walked up to a project root without a .scitex/dataset/ —
            # don't keep ascending into unrelated parents.
            return None
    return None


def runtime_dir() -> Path:
    """Runtime dir: project-scope if present, else user scope. Always exists on return."""
    root = project_root() or user_root()
    rt = root / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    return rt


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


@lru_cache(maxsize=8)
def _load_at(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return _load_yaml(p)


def find_config_path(cli_path: Optional[str] = None) -> Optional[Path]:
    """Resolve the config file via the SciTeX precedence chain."""
    if cli_path:
        p = Path(cli_path).expanduser()
        return p if p.is_file() else None

    env = os.environ.get(ENV_CONFIG)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    proj = project_root()
    if proj is not None:
        p = proj / "config.yaml"
        if p.is_file():
            return p

    p = user_root() / "config.yaml"
    if p.is_file():
        return p

    return None


def load_config(cli_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the resolved config, or an empty dict if none is found."""
    path = find_config_path(cli_path)
    if path is None:
        return {}
    return _load_at(str(path))


__all__ = [
    "PKG_SHORT",
    "ENV_CONFIG",
    "ENV_DIR",
    "user_root",
    "project_root",
    "runtime_dir",
    "find_config_path",
    "load_config",
]

# EOF
