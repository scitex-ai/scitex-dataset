#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/__init__.py

"""Agentic AI-for-science benchmarks.

This domain is categorically distinct from the raw catalog sources
(GEO / OpenNeuro / ChEMBL …) and on-demand sources (HuggingFace):
each member is an **agentic benchmark** whose preparation pipeline is
``download → prepare → mask`` rather than a single ``fetch``.

Benchmarks:
- ``corebench``        — CORE-Bench (90 reproducibility capsules, Princeton)
- ``bixbench``         — BixBench (205 bioinformatics tasks, FutureHouse)
- ``biomysterybench``  — BioMysteryBench (Anthropic; preview public,
                         full set gated)

On-disk contract (see ``_base.py``): one ``ai-for-science`` category dir
grouping every benchmark, each with two roles — ``raw/`` (upstream
snapshot stored as-is, operator-private, never mounted) and ``masked/``
(agent-visible, leak-safe view built by ``mask``). There is no separate
oracles tree: answers live in ``raw/`` and leak-prevention happens by
only ever mounting ``masked/``.

    <dataset-root>/ai-for-science/<benchmark>/{raw,masked,.scitex/dataset}

Every benchmark module exposes the same contract:

    def download(*, raw_dir: Path, **opts) -> dict:
        '''Fetch the upstream snapshot into raw_dir, as-is.'''

    def mask(*, raw_dir: Path, masked_dir: Path, **opts) -> dict:
        '''Build the leak-safe masked_dir view: oracle values nulled +
        answer-free upstream content symlinked.'''

    def prepare(*, dataset_root=None, **opts) -> dict:
        '''download + (optional inventory) + mask, then emit MANIFEST.yaml.'''

Each verb returns a dict describing what it did (paths written, record
counts, symlinks created). The dicts are JSON-serializable so they
round-trip cleanly through MCP / CLI ``--json`` output.

See ``_manifest.py`` for the canonical ``.scitex/dataset/MANIFEST.yaml``
schema (mirrors scitex-template's ``.scitex/template/MANIFEST.yaml``).
"""

from __future__ import annotations

from . import biomysterybench, bixbench, corebench
from ._base import DOMAIN, BenchmarkPaths, resolve_paths
from ._manifest import MANIFEST_FILENAME, ManifestEntry, write_manifest

__all__ = [
    # Submodules (the three benchmarks)
    "corebench",
    "bixbench",
    "biomysterybench",
    # Path helpers
    "DOMAIN",
    "BenchmarkPaths",
    "resolve_paths",
    # Manifest helpers
    "MANIFEST_FILENAME",
    "ManifestEntry",
    "write_manifest",
]

# EOF
