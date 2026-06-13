#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_base.py

"""Shared path resolution for agentic AI-for-science benchmarks.

One on-disk contract, two filesystem roles per benchmark — no separate
``oracles`` tree. Leak-prevention is handled *within* the dataset scope:

- ``raw_dir``    — upstream snapshot stored *as-is* (may contain answers,
                   rubrics, solutions). Operator-private: it is NEVER
                   bind-mounted into an agent capsule, and ``prepare``
                   leaves it read-only.
- ``masked_dir`` — agent-visible, leak-safe derivative. A curated view
                   built by ``mask``: symlinks to answer-free upstream
                   files plus transformed question files with the oracle
                   columns nulled. This is what the experiment harness
                   mounts read-only inside the agent capsule.
- ``manifest_dir`` — ``.scitex/dataset/`` holding ``MANIFEST.yaml``
                   (id + version + source + checksum + mask-seed), so a
                   single ``cp -r`` of the benchmark dir carries its
                   provenance along.

Canonical layout (one ``ai-for-science`` category dir grouping every
benchmark, mirroring the CLI grammar ``scitex-dataset ai-for-science
<benchmark> ...``)::

    <dataset-root>/ai-for-science/<benchmark>/
        raw/                      # upstream as-is (operator-private)
        masked/                   # agent-visible leak-safe view
        .scitex/dataset/MANIFEST.yaml

``<dataset-root>`` is resolved through :mod:`scitex_dataset._config`
(the same ``.scitex/dataset`` project→user convention every scitex
package uses), highest priority first:

1. explicit ``dataset_root=`` argument,
2. ``$SCITEX_DATASET_ROOT`` env override,
3. nearest project ``.scitex/dataset`` (``_config.project_root``),
4. per-user ``~/.scitex/dataset`` (``_config.user_root``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .._config import project_root, user_root

# Category directory grouping every AI-for-science benchmark under the
# dataset root — matches the CLI domain (``ai-for-science``).
DOMAIN = "ai-for-science"

# Established override honored before the _config project→user chain so
# the paper harness can point a prepared tree at a repo-local mount.
_DATASET_ROOT_ENV = "SCITEX_DATASET_ROOT"


def _dataset_root(override: Path | str | None = None) -> Path:
    """Resolve the dataset root (the dir that holds ``ai-for-science/``).

    Priority: explicit ``override`` > ``$SCITEX_DATASET_ROOT`` >
    nearest project ``.scitex/dataset`` > ``~/.scitex/dataset``.
    """
    if override is not None:
        return Path(override)
    env = os.environ.get(_DATASET_ROOT_ENV)
    if env:
        return Path(env)
    return project_root() or user_root()


@dataclass(frozen=True)
class BenchmarkPaths:
    """Resolved filesystem roles for one agentic benchmark."""

    benchmark: str
    root: Path
    raw_dir: Path
    masked_dir: Path
    manifest_dir: Path

    def as_dict(self) -> dict:
        """JSON-serialisable view (Path → str) for CLI / MCP output."""
        return {
            "benchmark": self.benchmark,
            "root": str(self.root),
            "raw_dir": str(self.raw_dir),
            "masked_dir": str(self.masked_dir),
            "manifest_dir": str(self.manifest_dir),
        }


def resolve_paths(
    benchmark: str,
    *,
    dataset_root: Path | str | None = None,
) -> BenchmarkPaths:
    """Resolve the canonical roles for ``benchmark``.

    ``benchmark`` is the bare benchmark name (``corebench``,
    ``bixbench``, ``biomysterybench``) — NOT a repo-specific cohort
    slug. The layout is::

        <dataset-root>/ai-for-science/<benchmark>/{raw,masked,.scitex/dataset}
    """
    root = _dataset_root(dataset_root) / DOMAIN / benchmark
    return BenchmarkPaths(
        benchmark=benchmark,
        root=root,
        raw_dir=root / "raw",
        masked_dir=root / "masked",
        manifest_dir=root / ".scitex" / "dataset",
    )


__all__ = [
    "DOMAIN",
    "BenchmarkPaths",
    "resolve_paths",
]

# EOF
