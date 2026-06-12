#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_base.py

"""Shared path resolution for agentic-benchmark cohorts.

Three filesystem roles, each addressable by env var so the same code
runs locally / in CI / on a SLURM compute node:

- ``oracle_dir``    — operator-private upstream-with-answers artifacts.
                      Default ``$SCITEX_ORACLES_ROOT/<cohort_dir>`` →
                      ``~/.scitex/oracles/<cohort_dir>``.
                      Never bind-mounted into an agent capsule.
- ``capsule_dir``   — bulk capsule code/data (multi-GB tarballs, HF
                      snapshots). Default
                      ``$SCITEX_DATASET_ROOT/<cohort_dir>/src/capsules``
                      → ``~/.scitex/dataset/<cohort_dir>/src/capsules``.
- ``benchmark_dir`` — agent-visible masked questions file. Default
                      ``$SCITEX_DATASET_ROOT/<cohort_dir>/src/benchmark``.
                      This is what the experiment harness mounts at
                      ``/questions:ro`` inside the agent capsule — must
                      contain ONLY masked derivatives (oracle answers
                      nulled).

Env vars honored (highest priority first):

- ``SCITEX_DATASET_ROOT`` — base dir for capsule + benchmark output.
- ``SCITEX_ORACLES_ROOT`` — base dir for operator-private oracle copies.

Each benchmark module names its own ``cohort_dir`` (e.g.
``cohort_a_corebench``) — those names match the existing on-disk
layout used by the paper repo, so a freshly-prepared dataset is
binary-compatible with consumers that already point at the legacy
paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Default roots — both lazily resolved so a test can ``monkeypatch`` the
# env vars before any benchmark module imports.
_DEFAULT_DATASET_ROOT_ENV = "SCITEX_DATASET_ROOT"
_DEFAULT_ORACLES_ROOT_ENV = "SCITEX_ORACLES_ROOT"


def _default_dataset_root() -> Path:
    return Path(
        os.environ.get(
            _DEFAULT_DATASET_ROOT_ENV,
            str(Path.home() / ".scitex" / "dataset"),
        )
    )


def _default_oracles_root() -> Path:
    return Path(
        os.environ.get(
            _DEFAULT_ORACLES_ROOT_ENV,
            str(Path.home() / ".scitex" / "oracles"),
        )
    )


# Sentinel default paths used by callers that just want the conventional
# layout — exposed at the package surface for documentation / smoke tests.
DEFAULT_ORACLE_DIR = _default_oracles_root()
DEFAULT_CAPSULE_DIR = _default_dataset_root()
DEFAULT_BENCHMARK_DIR = _default_dataset_root()


@dataclass(frozen=True)
class BenchmarkPaths:
    """Resolved filesystem roles for one agentic benchmark."""

    cohort_dir: str
    oracle_dir: Path
    capsule_dir: Path
    benchmark_dir: Path
    manifest_dir: Path

    def as_dict(self) -> dict:
        """JSON-serialisable view (Path → str) for CLI / MCP output."""
        return {
            "cohort_dir": self.cohort_dir,
            "oracle_dir": str(self.oracle_dir),
            "capsule_dir": str(self.capsule_dir),
            "benchmark_dir": str(self.benchmark_dir),
            "manifest_dir": str(self.manifest_dir),
        }


def resolve_paths(
    cohort_dir: str,
    *,
    oracle_root: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> BenchmarkPaths:
    """Resolve the four canonical roles for ``cohort_dir``.

    Optional overrides (``oracle_root`` / ``dataset_root``) take
    precedence over the env-var defaults. If both are None, env vars
    (``SCITEX_ORACLES_ROOT`` / ``SCITEX_DATASET_ROOT``) are consulted;
    failing that, the per-user ``~/.scitex/{oracles,dataset}`` defaults
    are used.

    The layout mirrors the existing paper-repo cohort tree so a freshly
    prepared dataset is byte-compatible with anything that still
    references ``data/<cohort_dir>/src/{capsules,benchmark}``.
    """
    oracle_root_p = (
        Path(oracle_root) if oracle_root is not None else _default_oracles_root()
    )
    dataset_root_p = (
        Path(dataset_root) if dataset_root is not None else _default_dataset_root()
    )

    cohort_root = dataset_root_p / cohort_dir
    return BenchmarkPaths(
        cohort_dir=cohort_dir,
        oracle_dir=oracle_root_p / cohort_dir,
        capsule_dir=cohort_root / "src" / "capsules",
        benchmark_dir=cohort_root / "src" / "benchmark",
        # Per-cohort provenance manifest — mirrors scitex-template's
        # ``.scitex/template/MANIFEST.yaml`` convention so a downstream
        # consumer can pin the exact snapshot (id + version + checksum +
        # mask-seed). Lives under the cohort root so a single
        # ``rsync``/``cp -r`` of the cohort directory carries it along.
        manifest_dir=cohort_root / ".scitex" / "dataset",
    )


__all__ = [
    "BenchmarkPaths",
    "DEFAULT_BENCHMARK_DIR",
    "DEFAULT_CAPSULE_DIR",
    "DEFAULT_ORACLE_DIR",
    "resolve_paths",
]

# EOF
