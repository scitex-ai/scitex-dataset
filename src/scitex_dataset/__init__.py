#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/__init__.py

"""
SciTeX Dataset - Unified interface for scientific dataset discovery.

Domains:
- neuroscience:    OpenNeuro, DANDI, PhysioNet
- general:         Scientific Data, Zenodo, Figshare, OpenML, HuggingFace
- biology:         GEO (Gene Expression Omnibus)
- pharmacology:    ChEMBL, MoleculeNet
- medical:         ClinicalTrials.gov
- ai-for-science:  CORE-Bench, BixBench, BioMysteryBench (agentic
                   reproducibility / bioinformatics / biology cohorts —
                   ``download | prepare | mask`` verb tree)

Usage:
    >>> from scitex_dataset import neuroscience
    >>> datasets = neuroscience.fetch_all_datasets(max_datasets=10)

    >>> # Or direct import for convenience
    >>> from scitex_dataset import fetch_all_datasets, search_datasets

    >>> # Local database for fast searching
    >>> from scitex_dataset import database as db
    >>> db.build()  # Fetch all sources and index
    >>> results = db.search("alzheimer EEG", min_subjects=20)

    >>> # Prepare an agentic benchmark (mask only — safe, fast)
    >>> from scitex_dataset import ai_for_science
    >>> paths = ai_for_science.resolve_paths("corebench")
    >>> ai_for_science.corebench.mask(
    ...     raw_dir=paths.raw_dir,
    ...     masked_dir=paths.masked_dir,
    ... )
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        __version__ = _v("scitex-dataset")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
    del _v, PackageNotFoundError
except ImportError:  # pragma: no cover — only on ancient Pythons
    __version__ = "0.0.0+local"
# Domain submodules
from . import _api as _api  # noqa: F401
from . import (
    ai_for_science,
    biology,
    database,
    general,
    medical,
    neuroscience,
    pharmacology,
)

# Per-source ``<src>_fetch`` / ``<src>_format`` aliases — give every MCP
# tool a matching Python callable for the audit-mcp-tools § 6 parity
# check. See _api.py for the explicit list.
from ._api import (  # noqa: F401
    biomysterybench_mask,
    bixbench_mask,
    chembl_fetch,
    clinicaltrials_fetch,
    corebench_mask,
    dandi_fetch,
    download_dataset,
    figshare_fetch,
    geo_fetch,
    gin_download,
    gin_fetch,
    gin_info,
    gin_search,
    huggingface_download_file,
    huggingface_fetch,
    huggingface_info,
    huggingface_search,
    moleculenet_fetch,
    openml_fetch,
    openneuro_fetch,
    physionet_fetch,
    zenodo_fetch,
)

# DB-level aliases for MCP parity (`dataset_db_*` tools).
from .database import build as db_build  # noqa: F401
from .database import get_stats as db_show_stats  # noqa: F401
from .database import search as db_search  # noqa: F401

# Convenience exports from neuroscience.openneuro (primary source)
from .neuroscience.openneuro import (
    OPENNEURO_API,
    fetch_all_datasets,
    fetch_datasets,
    format_dataset,
)
from .search import search_datasets, sort_datasets


def list_sources() -> dict:
    """Return the 11-source registry — matches ``dataset_list_sources`` MCP tool."""
    from ._sources import SOURCE_INFO

    return {"sources": SOURCE_INFO, "count": len(SOURCE_INFO)}


def filter_results(datasets, **kwargs):
    """Filter and rank dataset dicts — matches ``dataset_filter_results`` MCP tool."""
    from .search import search_datasets as _search
    from .search import sort_datasets as _sort

    sort_by = kwargs.pop("sort_by", "downloads")
    limit = kwargs.pop("limit", None)
    out = _search(datasets, **kwargs)
    out = _sort(out, by=sort_by, descending=True)
    return out[:limit] if limit else out


# Public Python API surface — kept in lock-step with the MCP tool set
# under `_mcp/_tools/` so `scitex-dev ecosystem audit-mcp-tools` § 6
# parity passes without skip_rules masking.
#
# The domain submodules (``neuroscience``, ``biology``, …) and the
# ``database`` module remain importable via ordinary attribute access
# (``scitex_dataset.biology.fetch_all_datasets``) — they are simply
# excluded from ``__all__`` so the audit doesn't flatten their
# convenience re-exports into a second, MCP-less Python surface. The
# bare convenience aliases ``fetch_*`` / ``format_dataset`` / ``search_*``
# / ``sort_*`` are likewise omitted: they are OpenNeuro-only shortcuts
# that masquerade as domain-level functions and have no matching MCP
# tool; the source-explicit ``openneuro_fetch`` is the supported entry.
__all__ = [
    "__version__",
    # Database aliases (MCP parity with ``dataset_db_*`` tools)
    "db_build",
    "db_search",
    "db_show_stats",
    # Search + filter (MCP parity with ``dataset_filter_results`` /
    # ``dataset_list_sources``)
    "filter_results",
    "list_sources",
    # Per-source ``<src>_fetch`` aliases (MCP parity with
    # ``dataset_<src>_fetch`` tools)
    "openneuro_fetch",
    "dandi_fetch",
    "physionet_fetch",
    "gin_fetch",
    "gin_search",
    "gin_info",
    "gin_download",
    "zenodo_fetch",
    "figshare_fetch",
    "openml_fetch",
    "moleculenet_fetch",
    "geo_fetch",
    "chembl_fetch",
    "clinicaltrials_fetch",
    # HuggingFace family
    "huggingface_fetch",
    "huggingface_search",
    "huggingface_info",
    "huggingface_download_file",
    # Unified dispatcher (issue #36)
    "download_dataset",
    # Agentic AI-for-science benchmark cohorts (MCP parity with
    # ``dataset_<bench>_mask`` tools)
    "corebench_mask",
    "bixbench_mask",
    "biomysterybench_mask",
]

# EOF
