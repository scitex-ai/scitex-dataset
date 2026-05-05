#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-30 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/_branding.py

"""Branding module for scitex-dataset.

This module provides consistent branding and naming conventions
for the scitex-dataset package, including environment variable prefixes
and MCP server configuration.
"""

from __future__ import annotations

import os
from typing import Optional

# Package identification
PACKAGE_NAME = "scitex-dataset"
PACKAGE_DISPLAY_NAME = "SciTeX Dataset"
PACKAGE_DESCRIPTION = "Unified access to scientific datasets for AI-powered research"

# Environment variable prefix
ENV_PREFIX = "SCITEX_DATASET"

# MCP configuration
MCP_SERVER_NAME = "scitex-dataset"
MCP_TOOL_PREFIX = "dataset"


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with package prefix.

    Parameters
    ----------
    key : str
        Variable name without prefix (e.g., "CACHE_DIR").
    default : str, optional
        Default value if not set.

    Returns
    -------
    str or None
        Environment variable value.

    Example
    -------
    >>> get_env("CACHE_DIR", "/tmp/cache")
    # Reads SCITEX_DATASET_CACHE_DIR or returns "/tmp/cache"
    """
    return os.environ.get(f"{ENV_PREFIX}_{key}", default)


def get_mcp_server_name() -> str:
    """Get the MCP server name."""
    return MCP_SERVER_NAME


def get_mcp_instructions() -> str:
    """Get MCP server instructions for LLM context."""
    return """SciTeX Dataset: Unified access to scientific datasets.

This MCP server provides tools to discover and query datasets from major
scientific repositories:

**Repositories:**
- OpenNeuro: BIDS-formatted neuroimaging (MRI, EEG, MEG, iEEG, PET)
- DANDI Archive: NWB neurophysiology data
- PhysioNet: EEG, ECG, and physiological signals
- Zenodo: General scientific data repository (CERN)

**Tools:**
- dataset_openneuro_fetch: Fetch OpenNeuro datasets
- dataset_dandi_fetch: Fetch DANDI dandisets
- dataset_physionet_fetch: Fetch PhysioNet databases
- dataset_zenodo_fetch: Fetch Zenodo datasets
- dataset_filter_results: Filter / rank fetched datasets in memory
- dataset_list_sources: List available data sources
- dataset_db_build: Build local search database
- dataset_db_search: Full-text search local database
- dataset_db_stats: Database statistics

**Typical Workflow:**
1. Fetch datasets: dataset_openneuro_fetch(max_datasets=100)
2. Filter results: dataset_filter_results(datasets, modality="eeg", min_subjects=20)
3. Or build local DB: dataset_db_build() then dataset_db_search("alzheimer EEG")

**Tips:**
- Use max_datasets to limit API calls during exploration
- Build local database for faster repeated searches
- Combine modality and subject filters for targeted results
"""


# EOF
