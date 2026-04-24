#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/medical/__init__.py

"""
Medical dataset sources.

Platforms:
- clinicaltrials: ClinicalTrials.gov (clinical studies, interventional trials)
"""

from . import clinicaltrials
from .clinicaltrials import (
    CLINICALTRIALS_API,
    fetch_all_datasets,
    fetch_datasets,
    format_dataset,
)

__all__ = [
    "clinicaltrials",
    "CLINICALTRIALS_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]

# EOF
