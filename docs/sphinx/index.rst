.. scitex-dataset documentation master file

scitex-dataset - Multi-domain Scientific Dataset Fetcher
=========================================================

**scitex-dataset** is a unified Python interface for discovering and fetching scientific datasets across **11 repositories** — neuroscience (OpenNeuro, DANDI, PhysioNet), general science (Zenodo, Figshare, OpenML, HuggingFace Hub), biology (GEO), pharmacology (MoleculeNet, ChEMBL), and medical (ClinicalTrials.gov).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   cli_reference
   sources

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/scitex_dataset

Key Features
------------

- **Unified Interface**: Single API across 10 catalog sources + HuggingFace Hub (on-demand)
- **Multi-domain**: Neuroscience, biology, pharmacology, medical, and general science
- **Local SQLite + FTS5 Index**: Build an offline-searchable cache via ``db_build()``
- **CLI & Python API**: Use from command line or import as a library
- **MCP Integration**: AI agents can discover and fetch datasets via MCP server

Supported Sources
-----------------

*Neuroscience*
- **OpenNeuro**: BIDS-formatted neuroimaging (MRI, EEG, MEG, iEEG, PET)
- **DANDI Archive**: NWB-formatted neurophysiology data
- **PhysioNet**: Physiological signal databases (ECG, EEG, waveforms)

*General Science*
- **Zenodo**: General scientific data repository (CERN)
- **Figshare**: Research data sharing platform
- **OpenML**: Machine learning datasets and benchmarks
- **HuggingFace Hub**: ML datasets and models (on-demand)

*Biology*
- **GEO**: Gene Expression Omnibus (NCBI) — transcriptomics, microarray

*Pharmacology*
- **MoleculeNet**: Molecular ML benchmark suite
- **ChEMBL**: Bioactivity database (EBI) — IC50/Ki/EC50 assays

*Medical*
- **ClinicalTrials.gov**: NIH clinical study registry

Quick Example
-------------

Python API:

.. code-block:: python

    from scitex_dataset import openneuro_fetch, filter_results, db_build, db_search

    # Fetch from OpenNeuro
    records = openneuro_fetch(max_datasets=100)

    # Filter + rank in memory
    top = filter_results(records, modality="eeg", min_subjects=20, sort_by="downloads", limit=10)

    # Build the local SQLite + FTS5 index for offline queries
    db_build()
    results = db_search("Alzheimer EEG")

CLI:

.. code-block:: bash

    # Fetch datasets
    scitex-dataset neuroscience openneuro fetch -n 50

    # Search for datasets
    scitex-dataset general huggingface search "biology" -n 20

    # Build/search the local database
    scitex-dataset db build
    scitex-dataset db search "Alzheimer EEG"

    # List available sources (embedded in --help)
    scitex-dataset --help

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
