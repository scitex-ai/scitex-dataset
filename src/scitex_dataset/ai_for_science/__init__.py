#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/__init__.py

"""Agentic AI-for-science benchmarks.

This domain is categorically distinct from the raw catalog sources
(GEO / OpenNeuro / ChEMBL …) and on-demand sources (HuggingFace):
each member is an **agentic benchmark** whose preparation pipeline is
``download → prepare → standardize`` rather than a single ``fetch``.

Benchmarks:
- ``corebench``        — CORE-Bench (90 reproducibility capsules, Princeton)
- ``bixbench``         — BixBench (205 bioinformatics tasks, FutureHouse)
- ``biomysterybench``  — BioMysteryBench (Anthropic; preview public,
                         full set gated)

On-disk contract (see ``_base.py``): one ``ai-for-science`` category dir
grouping every benchmark, each with three roles — ``raw/`` (upstream
snapshot stored as-is, operator-private, never mounted), ``for_solver/``
(agent-visible, answer-masked UNIFORM view built by ``standardize``), and
``eval/`` (operator-side scorer view, never mounted). Answers live in
``raw/`` + ``eval/``. Bind only the selected ``for_solver/capsule-NNN/``;
existing masking rules are not a qualification of arbitrary real inputs.
The root catalog and private materialization ledger stay outside that bind.

    <dataset-root>/ai-for-science/<benchmark>/{raw,for_solver,eval,.scitex/dataset}

BioMysteryBench full uses separate raw and output variant namespaces;
consume the paths returned by ``prepare(download_full=True)``.

Every benchmark module exposes the same contract:

    def download(*, raw_dir: Path, **opts) -> dict:
        '''Fetch the upstream snapshot into raw_dir, as-is.'''

    def standardize(*, raw_dir, for_solver_dir, eval_dir, **opts) -> dict:
        '''Split raw oracle into an answer-masked per-capsule task view
        (task.jsonl + submission schema) and the private operator eval/ view
        (answers.jsonl + evaluate.py).'''

    def prepare(*, dataset_root=None, **opts) -> dict:
        '''download + (optional inventory) + standardize, then emit
        MANIFEST.yaml.'''

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
from ._score import score_submission
from ._standardize import (
    UNIFORM_SUBMISSION_SCHEMA,
    write_eval,
    write_for_solver,
)
from ._validate import validate_submission

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
    # Standardize helpers
    "UNIFORM_SUBMISSION_SCHEMA",
    "write_for_solver",
    "write_eval",
    # Submission validation + scoring
    "validate_submission",
    "score_submission",
]

# EOF
