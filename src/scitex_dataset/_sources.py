#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_sources.py

"""Single source-of-truth for the source registry.

Two lists exist on purpose:

- ``CATALOG_SOURCES`` — sources whose modules expose the standard
  ``fetch_all_datasets() / format_dataset()`` contract and can be
  enumerated/indexed wholesale.
- ``ONDEMAND_SOURCES`` — sources fetched by repo_id (no full catalog
  enumeration). HuggingFace is the only one today.

``ALL_SOURCES = CATALOG_SOURCES + ONDEMAND_SOURCES`` is the union used
by CLI choice lists and ``dataset_list_sources``.
"""

CATALOG_SOURCES = [
    "openneuro",
    "dandi",
    "physionet",
    "zenodo",
    "figshare",
    "openml",
    "geo",
    "chembl",
    "moleculenet",
    "clinicaltrials",
]

ONDEMAND_SOURCES = [
    "huggingface",
]

ALL_SOURCES = CATALOG_SOURCES + ONDEMAND_SOURCES

# Catalog metadata used by `dataset_list_sources` MCP tool.
SOURCE_INFO = {
    "openneuro": {
        "name": "OpenNeuro",
        "description": "BIDS neuroimaging (MRI, EEG, MEG, iEEG, PET)",
        "url": "https://openneuro.org",
        "format": "BIDS",
        "domain": "neuroscience",
        "kind": "catalog",
    },
    "dandi": {
        "name": "DANDI Archive",
        "description": "NWB neurophysiology data",
        "url": "https://dandiarchive.org",
        "format": "NWB",
        "domain": "neuroscience",
        "kind": "catalog",
    },
    "physionet": {
        "name": "PhysioNet",
        "description": "EEG, ECG, physiological signals",
        "url": "https://physionet.org",
        "format": "Various",
        "domain": "neuroscience",
        "kind": "catalog",
    },
    "zenodo": {
        "name": "Zenodo",
        "description": "General scientific data repository (CERN)",
        "url": "https://zenodo.org",
        "format": "Various",
        "domain": "general",
        "kind": "catalog",
    },
    "figshare": {
        "name": "Figshare",
        "description": "General research data sharing platform",
        "url": "https://figshare.com",
        "format": "Various",
        "domain": "general",
        "kind": "catalog",
    },
    "openml": {
        "name": "OpenML",
        "description": "Machine learning datasets and benchmarks",
        "url": "https://www.openml.org",
        "format": "ARFF/CSV/Parquet",
        "domain": "machine-learning",
        "kind": "catalog",
    },
    "moleculenet": {
        "name": "MoleculeNet",
        "description": "Molecular ML benchmark suite",
        "url": "https://moleculenet.org",
        "format": "CSV/SDF",
        "domain": "pharmacology",
        "kind": "catalog",
    },
    "geo": {
        "name": "GEO",
        "description": "Gene Expression Omnibus (NCBI)",
        "url": "https://www.ncbi.nlm.nih.gov/geo/",
        "format": "SOFT/MINiML",
        "domain": "biology",
        "kind": "catalog",
    },
    "chembl": {
        "name": "ChEMBL",
        "description": "Bioactivity database (EBI)",
        "url": "https://www.ebi.ac.uk/chembl/",
        "format": "Various",
        "domain": "pharmacology",
        "kind": "catalog",
    },
    "clinicaltrials": {
        "name": "ClinicalTrials.gov",
        "description": "Clinical study registry (NIH)",
        "url": "https://clinicaltrials.gov",
        "format": "JSON/XML",
        "domain": "medical",
        "kind": "catalog",
    },
    "huggingface": {
        "name": "HuggingFace Hub",
        "description": "ML datasets and models — fetched on demand by repo_id (not catalog-enumerated)",
        "url": "https://huggingface.co",
        "format": "Various",
        "domain": "machine-learning",
        "kind": "ondemand",
    },
}

DOMAIN_OF = {
    "openneuro": "neuroscience",
    "dandi": "neuroscience",
    "physionet": "neuroscience",
    "zenodo": "general",
    "figshare": "general",
    "openml": "general",
    "moleculenet": "pharmacology",
    "geo": "biology",
    "chembl": "pharmacology",
    "clinicaltrials": "medical",
    "huggingface": "general",
}

DOMAINS = ["neuroscience", "general", "biology", "pharmacology", "medical"]


def sources_in_domain(domain: str):
    """Return the source ids that live under ``domain``."""
    return [s for s, d in DOMAIN_OF.items() if d == domain]


__all__ = [
    "CATALOG_SOURCES",
    "ONDEMAND_SOURCES",
    "ALL_SOURCES",
    "SOURCE_INFO",
    "DOMAIN_OF",
    "DOMAINS",
    "sources_in_domain",
]

# EOF
