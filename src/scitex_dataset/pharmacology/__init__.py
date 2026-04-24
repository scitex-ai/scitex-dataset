#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/pharmacology/__init__.py

"""
Pharmacology dataset sources.

Platforms:
- chembl: ChEMBL bioactivity database (drug discovery, assays)
- moleculenet: MoleculeNet benchmark suite for molecular ML
"""

from . import chembl, moleculenet

__all__ = [
    "chembl",
    "moleculenet",
]

# EOF
