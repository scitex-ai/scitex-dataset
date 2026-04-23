#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP tools registration for scitex-dataset.

This module provides a function to register all scitex-dataset MCP tools
with an external FastMCP server (e.g., scitex-python's unified MCP server).
"""

from typing import Any, Dict, List, Optional


def register_all_tools(mcp) -> None:
    """Register all scitex-dataset tools with an MCP server.

    Parameters
    ----------
    mcp : FastMCP
        The FastMCP server instance to register tools with.
    """

    # OpenNeuro
    @mcp.tool()
    def dataset_openneuro_fetch(
        max_datasets: int = 100,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch BIDS dataset metadata (MRI / EEG / MEG / iEEG / PET) from OpenNeuro via its GraphQL API and normalize into a uniform dict (id, name, subjects, modality, task, downloads, readme). Drop-in replacement for `openneuro-python` + hand-rolled GraphQL queries against `https://openneuro.org/crn/graphql`. Use when the user asks to "list OpenNeuro datasets", "get BIDS datasets", "find MRI/EEG/MEG studies on OpenNeuro", or is building a dataset index."""
        from ..neuroscience.openneuro import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            batch_size=batch_size,
            max_datasets=max_datasets if max_datasets > 0 else None,
        )
        return [format_dataset(ds) for ds in raw]

    # DANDI
    @mcp.tool()
    def dataset_dandi_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch NWB neurophysiology dataset metadata from DANDI Archive (https://dandiarchive.org) and normalize into a uniform dict. Drop-in replacement for `dandi` CLI / `dandischema` + raw REST queries against the DANDI API. Use when the user asks to "list DANDI dandisets", "find Neuropixels / Allen Brain / NWB datasets", "search DANDI for spike sorting / calcium imaging", or is indexing neurophysiology data."""
        from ..neuroscience.dandi import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    # PhysioNet
    @mcp.tool()
    def dataset_physionet_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch physiological signal dataset metadata (EEG, ECG, EMG, PPG, polysomnography, ICU waveforms) from PhysioNet (https://physionet.org) and normalize. Drop-in replacement for `wfdb` index scraping + raw `requests` against PhysioNet's project listing. Use when the user asks to "list PhysioNet datasets", "find ECG / sleep / ICU datasets", "search MIMIC / SHHS / Sleep-EDF", or is indexing clinical physiology data."""
        from ..neuroscience.physionet import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    # GEO
    @mcp.tool()
    def dataset_geo_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch transcriptomics / genomics dataset metadata (GSE series, GDS, platforms) from NCBI GEO (Gene Expression Omnibus) and normalize. Drop-in replacement for `GEOparse` + raw E-utilities (esearch / esummary) queries. Use when the user asks to "find GEO series on Alzheimer's / cancer", "list RNA-seq / microarray datasets", "search gene-expression studies", or mentions GSE / GDS accessions."""
        from ..biology.geo import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    # ChEMBL
    @mcp.tool()
    def dataset_chembl_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch bioactivity / drug-discovery dataset metadata (targets, assays, compounds, IC50 / Ki / EC50) from ChEMBL (EBI) and normalize. Drop-in replacement for `chembl-webresource-client` + raw REST queries against `https://www.ebi.ac.uk/chembl/api/data`. Use when the user asks to "find ChEMBL targets / assays", "search bioactivity data", "list drug-discovery datasets", or mentions CHEMBL IDs, pIC50, ligand efficiency."""
        from ..pharmacology.chembl import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    # ClinicalTrials
    @mcp.tool()
    def dataset_clinicaltrials_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch clinical-study metadata (interventional trials, observational studies, outcomes, phases, status) from ClinicalTrials.gov v2 API and normalize. Drop-in replacement for `pytrials` + raw REST queries against `https://clinicaltrials.gov/api/v2/studies`. Use when the user asks to "find clinical trials on X", "list Phase III studies for disease Y", "search ClinicalTrials.gov", or mentions NCT IDs, interventions, trial phase."""
        from ..medical.clinicaltrials import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    # Search
    @mcp.tool()
    def dataset_search(
        datasets: List[Dict[str, Any]],
        modality: Optional[str] = None,
        min_subjects: Optional[int] = None,
        max_subjects: Optional[int] = None,
        task_contains: Optional[str] = None,
        text_query: Optional[str] = None,
        min_downloads: Optional[int] = None,
        has_readme: bool = False,
        sort_by: str = "downloads",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Filter + rank a list of normalized dataset dicts (from any source) by modality / subject count / task / full-text / downloads / readme presence, then sort. Drop-in replacement for hand-rolled list comprehensions + `sorted(key=...)` over the fetcher output. Use when the user asks to "find EEG datasets with >50 subjects", "rank DANDI by downloads", "filter OpenNeuro by task='rest'", "search across sources", or is chaining a fetcher into a filter step."""
        from ..search import search_datasets, sort_datasets

        results = search_datasets(
            datasets,
            modality=modality,
            min_subjects=min_subjects,
            max_subjects=max_subjects,
            task_contains=task_contains,
            text_query=text_query,
            min_downloads=min_downloads,
            has_readme=has_readme,
        )
        results = sort_datasets(results, by=sort_by, descending=True)
        return results[:limit]

    # List sources
    @mcp.tool()
    def dataset_list_sources() -> Dict[str, Any]:
        """Enumerate every supported dataset repository (OpenNeuro, DANDI, PhysioNet, Zenodo, GEO, ChEMBL, ClinicalTrials) with its canonical URL, file format (BIDS / NWB / Various / JSON), and domain (neuroscience / general / biology / pharmacology / medical). Use when the user asks "what sources can I search?", "which repositories are supported?", or before calling a per-source fetcher."""
        return {
            "sources": {
                "openneuro": {
                    "name": "OpenNeuro",
                    "description": "BIDS neuroimaging (MRI, EEG, MEG, iEEG, PET)",
                    "url": "https://openneuro.org",
                    "format": "BIDS",
                    "domain": "neuroscience",
                },
                "dandi": {
                    "name": "DANDI Archive",
                    "description": "NWB neurophysiology data",
                    "url": "https://dandiarchive.org",
                    "format": "NWB",
                    "domain": "neuroscience",
                },
                "physionet": {
                    "name": "PhysioNet",
                    "description": "EEG, ECG, physiological signals",
                    "url": "https://physionet.org",
                    "format": "Various",
                    "domain": "neuroscience",
                },
                "zenodo": {
                    "name": "Zenodo",
                    "description": "General scientific data repository (CERN)",
                    "url": "https://zenodo.org",
                    "format": "Various",
                    "domain": "general",
                },
                "geo": {
                    "name": "GEO",
                    "description": "Gene Expression Omnibus (genomics, transcriptomics)",
                    "url": "https://www.ncbi.nlm.nih.gov/geo/",
                    "format": "Various",
                    "domain": "biology",
                },
                "chembl": {
                    "name": "ChEMBL",
                    "description": "Bioactivity database for drug discovery",
                    "url": "https://www.ebi.ac.uk/chembl/",
                    "format": "JSON",
                    "domain": "pharmacology",
                },
                "clinicaltrials": {
                    "name": "ClinicalTrials.gov",
                    "description": "Clinical study records (interventional trials)",
                    "url": "https://clinicaltrials.gov",
                    "format": "JSON",
                    "domain": "medical",
                },
            },
            "count": 7,
        }

    # Database tools
    @mcp.tool()
    def dataset_db_build(
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Crawl every (or selected) dataset repository and populate a local SQLite + FTS5 index for instant offline search. One-shot setup; re-run to refresh. Drop-in replacement for re-calling every fetcher every time + pandas in-memory filtering. Use when the user asks to "build the dataset index", "rebuild my local dataset DB", "refresh OpenNeuro locally", or before heavy `dataset_db_search` usage."""
        from .. import database

        counts = database.build(sources=sources)
        stats = database.get_stats()

        return {
            "success": True,
            "indexed": counts,
            "total": sum(counts.values()),
            "database_path": stats.get("path"),
            "size_mb": stats.get("size_mb"),
        }

    @mcp.tool()
    def dataset_db_search(
        query: Optional[str] = None,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        min_subjects: Optional[int] = None,
        max_subjects: Optional[int] = None,
        min_downloads: Optional[int] = None,
        has_readme: bool = False,
        limit: int = 20,
        order_by: str = "downloads",
    ) -> List[Dict[str, Any]]:
        """Query the local SQLite+FTS5 dataset index — full-text search across names/descriptions/tasks + structured filters (source, modality, subject-count range, downloads, readme) + configurable ORDER BY. Offline, fast, no network. Use when the user asks to "search my local dataset DB", "find EEG datasets offline", "query cached OpenNeuro+DANDI together", or is iterating search interactively. Requires `dataset_db_build` first."""
        from .. import database

        return database.search(
            query=query,
            source=source,
            modality=modality,
            min_subjects=min_subjects,
            max_subjects=max_subjects,
            min_downloads=min_downloads,
            has_readme=has_readme,
            limit=limit,
            order_by=order_by,
        )

    @mcp.tool()
    def dataset_db_stats() -> Dict[str, Any]:
        """Report local dataset-index health — per-source row counts, total size in MB, DB path, last-built timestamp. Use when the user asks "how many datasets are indexed?", "is my local DB up-to-date?", "show DB stats", or is diagnosing an empty `dataset_db_search` result."""
        from .. import database

        return database.get_stats()


# EOF
