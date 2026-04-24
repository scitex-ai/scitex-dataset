#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/__init__.py

"""
SciTeX Dataset - Unified interface for scientific dataset discovery.

Domains:
- neuroscience: OpenNeuro, DANDI, PhysioNet
- general: Scientific Data, Zenodo
- biology: GEO (Gene Expression Omnibus)
- pharmacology: ChEMBL
- medical: ClinicalTrials.gov

Usage:
    >>> from scitex_dataset import neuroscience
    >>> datasets = neuroscience.fetch_all_datasets(max_datasets=10)

    >>> # Or direct import for convenience
    >>> from scitex_dataset import fetch_all_datasets, search_datasets

    >>> # Local database for fast searching
    >>> from scitex_dataset import database as db
    >>> db.build()  # Fetch all sources and index
    >>> results = db.search("alzheimer EEG", min_subjects=20)
"""

__version__ = "0.3.1"

# Domain submodules
from . import biology, database, general, medical, neuroscience, pharmacology

# Convenience exports from neuroscience.openneuro (primary source)
from .neuroscience.openneuro import (
    OPENNEURO_API,
    fetch_all_datasets,
    fetch_datasets,
    format_dataset,
)
from .search import search_datasets, sort_datasets

__all__ = [
    "__version__",
    # Domains
    "neuroscience",
    "general",
    "biology",
    "pharmacology",
    "medical",
    # Database
    "database",
    # Convenience (OpenNeuro)
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
    "OPENNEURO_API",
    # Search
    "search_datasets",
    "sort_datasets",
]

# EOF
