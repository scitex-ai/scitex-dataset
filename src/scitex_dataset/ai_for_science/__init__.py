#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/__init__.py

"""Agentic AI-for-science benchmark cohorts.

This domain is categorically distinct from the raw catalog sources
(GEO / OpenNeuro / ChEMBL …) and on-demand sources (HuggingFace):
each member is an **agentic benchmark** whose preparation pipeline is
``download → prepare → mask`` rather than a single ``fetch``.

Benchmarks:
- ``corebench``        — CORE-Bench (90 reproducibility capsules, Princeton)
- ``bixbench``         — BixBench (205 bioinformatics tasks, FutureHouse)
- ``biomysterybench``  — BioMysteryBench (Anthropic; preview public,
                         full set gated)

Migration boundary (operator brief, 2026-06-12):
- The **dataset-prep** half migrates here from
  ``paper-scitex-clew/scripts/cohorts/{a,b,c}/dataset/``.
- The **experiment harness** (capsule launcher, SAC manifest, verifier,
  prompts) stays in the paper repo under
  ``scripts/cohorts/_shared/{launcher,spec,verify,prompts}``.

Every benchmark module exposes the same contract:

    def download(*, oracle_dir: Path, capsule_dir: Path, **opts) -> dict:
        '''Fetch upstream artifacts. Side-effect: writes oracle_dir + capsule_dir.'''

    def mask(*, oracle_dir: Path, benchmark_dir: Path, **opts) -> dict:
        '''Read oracle artifacts, write masked agent-visible questions file.'''

    def prepare(*, oracle_dir: Path, capsule_dir: Path,
                benchmark_dir: Path, manifest_dir: Path, **opts) -> dict:
        '''download + (optional inventory) + mask, then emit MANIFEST.yaml.'''

Each verb returns a dict describing what it did (paths written, record
counts, mask-seed where applicable). The dicts are JSON-serializable so
they round-trip cleanly through MCP / CLI ``--json`` output.

See ``_base.py`` for the shared path resolver + provenance manifest
helpers and ``_manifest.py`` for the canonical
``.scitex/dataset/MANIFEST.yaml`` schema (mirrors scitex-template's
``.scitex/template/MANIFEST.yaml``).
"""

from __future__ import annotations

from . import biomysterybench, bixbench, corebench
from ._base import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_CAPSULE_DIR,
    DEFAULT_ORACLE_DIR,
    BenchmarkPaths,
    resolve_paths,
)
from ._manifest import MANIFEST_FILENAME, ManifestEntry, write_manifest

__all__ = [
    # Submodules (the three benchmarks)
    "corebench",
    "bixbench",
    "biomysterybench",
    # Path helpers
    "BenchmarkPaths",
    "DEFAULT_BENCHMARK_DIR",
    "DEFAULT_CAPSULE_DIR",
    "DEFAULT_ORACLE_DIR",
    "resolve_paths",
    # Manifest helpers
    "MANIFEST_FILENAME",
    "ManifestEntry",
    "write_manifest",
]

# EOF
