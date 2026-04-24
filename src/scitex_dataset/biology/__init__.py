#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/biology/__init__.py

"""
Biology dataset sources.

Platforms:
- geo: Gene Expression Omnibus (genomics, transcriptomics)
"""

from . import geo
from .geo import (
    GEO_API,
    fetch_all_datasets,
    fetch_datasets,
    format_dataset,
)

__all__ = [
    "geo",
    "GEO_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]

# EOF
