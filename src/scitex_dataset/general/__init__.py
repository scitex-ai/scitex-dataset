#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-30 10:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/general/__init__.py

"""
General scientific dataset sources.

Platforms:
- zenodo: CERN's general scientific repository
- figshare: Research data sharing
- openml: Machine learning dataset repository
- huggingface: HF Hub, fetched on demand by repo id
- reference: well-known reference files (MNIST, ...) pinned by sha256
"""

from . import figshare, huggingface, openml, reference, zenodo

__all__ = [
    "zenodo",
    "figshare",
    "openml",
    "huggingface",
    "reference",
]

# EOF
