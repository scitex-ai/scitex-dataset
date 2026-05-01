#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/_mcp/server.py

"""
MCP server for scitex-dataset - unified scientific dataset discovery.

Usage:
    fastmcp run scitex_dataset._mcp.server:mcp
    # or
    scitex-dataset mcp start
"""

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .._branding import get_mcp_instructions, get_mcp_server_name

mcp = FastMCP(
    name=get_mcp_server_name(),
    instructions=get_mcp_instructions(),
)


@mcp.tool
def dataset_openneuro_fetch(
    max_datasets: int = 100,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch BIDS neuroimaging dataset metadata from OpenNeuro - use whenever the user asks to find, list, or download MRI/fMRI/EEG/MEG/iEEG/PET datasets, search OpenNeuro accessions (ds004xxx), or browse BIDS-formatted neuroimaging data. Drop-in replacement for openneuro-py, raw DataLad calls, or direct OpenNeuro GraphQL/API queries.

    OpenNeuro hosts BIDS-formatted neuroimaging data including
    MRI, EEG, MEG, iEEG, and PET.

    Parameters
    ----------
    max_datasets : int
        Maximum datasets to fetch (0 for all).
    batch_size : int
        Datasets per API request.

    Returns
    -------
    list
        List of formatted dataset dictionaries with fields:
        id, name, n_subjects, modalities, tasks, size_gb, downloads, readme, etc.
    """
    from ..neuroscience.openneuro import fetch_all_datasets, format_dataset

    raw = fetch_all_datasets(
        batch_size=batch_size,
        max_datasets=max_datasets if max_datasets > 0 else None,
    )
    return [format_dataset(ds) for ds in raw]


@mcp.tool
def dataset_dandi_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
    """Fetch DANDI Archive dataset metadata - use whenever the user asks to find, list, or download electrophysiology, neurophysiology, NWB-format, or optical imaging datasets from DANDI (dandiset IDs). Drop-in replacement for dandi-cli, pynwb-based DANDI downloads, or direct DANDI REST API calls.

    DANDI hosts neurophysiology data in NWB format.

    Parameters
    ----------
    max_datasets : int
        Maximum datasets to fetch (0 for all).

    Returns
    -------
    list
        List of formatted dandiset dictionaries.
    """
    from ..neuroscience.dandi import fetch_all_datasets, format_dataset

    raw = fetch_all_datasets(max_datasets=max_datasets if max_datasets > 0 else None)
    return [format_dataset(ds) for ds in raw]


@mcp.tool
def dataset_physionet_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
    """Fetch PhysioNet dataset metadata - use whenever the user asks to find, list, or download ECG, EEG, EMG, clinical waveforms, MIT-BIH arrhythmia, MIMIC waveforms, or other physiological signal databases from PhysioNet. Drop-in replacement for wfdb-python database listings or direct PhysioNet website downloads.

    PhysioNet hosts physiological signal databases including
    EEG, ECG, EMG, and other biomedical signals.

    Parameters
    ----------
    max_datasets : int
        Maximum datasets to fetch (0 for all).

    Returns
    -------
    list
        List of formatted database dictionaries.
    """
    from ..neuroscience.physionet import fetch_all_datasets, format_dataset

    raw = fetch_all_datasets(max_datasets=max_datasets if max_datasets > 0 else None)
    return [format_dataset(ds) for ds in raw]


@mcp.tool
def dataset_zenodo_fetch(
    query: str = "",
    max_datasets: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch Zenodo research dataset metadata - use whenever the user asks to find, list, or resolve research data by DOI, query a Zenodo record, or discover open-access datasets/software/publications from CERN's Zenodo repository. Drop-in replacement for zenodopy, raw requests calls to the Zenodo REST API, or manual DOI resolution.

    Zenodo is CERN's general-purpose open repository for research data,
    software, publications, and other research artifacts.

    Parameters
    ----------
    query : str
        Search query string (optional).
    max_datasets : int
        Maximum datasets to fetch (0 for all).

    Returns
    -------
    list
        List of formatted dataset dictionaries with fields:
        id, name, doi, authors, keywords, size_gb, downloads, etc.
    """
    from ..general.zenodo import fetch_all_datasets, format_dataset

    raw = fetch_all_datasets(
        query=query,
        max_datasets=max_datasets if max_datasets > 0 else None,
    )
    return [format_dataset(ds) for ds in raw]


@mcp.tool
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
    """Search and filter dataset metadata by modality/subjects/task/downloads/text - use whenever the user asks to find datasets matching criteria (e.g. "EEG datasets with >30 subjects", "MRI motor task studies") across already-fetched OpenNeuro/DANDI/PhysioNet/Zenodo results. Drop-in replacement for scraping each repository separately or writing custom pandas filters on raw metadata.

    Parameters
    ----------
    datasets : list
        List of dataset dictionaries (from fetch tools).
    modality : str, optional
        Filter by modality (mri, eeg, meg, etc.).
    min_subjects : int, optional
        Minimum number of subjects.
    max_subjects : int, optional
        Maximum number of subjects.
    task_contains : str, optional
        Filter by task name substring.
    text_query : str, optional
        Search in name and readme text.
    min_downloads : int, optional
        Minimum download count.
    has_readme : bool
        Only include datasets with readme.
    sort_by : str
        Sort by: downloads, views, n_subjects, size_gb.
    limit : int
        Maximum results to return.

    Returns
    -------
    list
        Filtered and sorted datasets.
    """
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


@mcp.tool
def dataset_list_sources() -> Dict[str, Any]:
    """List dataset repositories supported by scitex-dataset - use whenever the user asks what dataset sources/repositories/archives are available, which scientific data hubs scitex-dataset knows about, or which domains (neuroscience, general research) are covered. Returns OpenNeuro, DANDI, PhysioNet, Zenodo with URLs, formats, and domain tags.

    Returns
    -------
    dict
        Dictionary with source names and descriptions.
    """
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
        },
        "count": 4,
    }


# Database tools
@mcp.tool
def dataset_db_build(
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build or rebuild the local SQLite dataset index - use whenever the user asks to index dataset metadata locally for fast offline search, refresh the scitex-dataset cache, or populate the full-text search database across OpenNeuro/DANDI/PhysioNet/Zenodo. Drop-in replacement for manually pickling fetch results or writing a custom SQLite indexer.

    Fetches metadata from all sources and indexes them in a local
    SQLite database for fast full-text searching.

    Parameters
    ----------
    sources : list, optional
        Sources to index: ["openneuro", "dandi", "physionet", "zenodo"].
        Default: all sources.

    Returns
    -------
    dict
        Build results with counts per source.
    """
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


@mcp.tool
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
    """Search the local dataset database with SQLite FTS5 full-text search - use whenever the user asks to search datasets offline by keyword/modality/source without hitting remote APIs, query the scitex-dataset index, or filter indexed records by subjects/downloads/readme. Requires dataset_db_build to have run first. Drop-in replacement for grepping cached JSON or repeating remote API calls.

    Requires db_build to have been run first.

    Parameters
    ----------
    query : str, optional
        Full-text search query (searches name, readme, tasks).
    source : str, optional
        Filter by source: "openneuro", "dandi", "physionet", "zenodo".
    modality : str, optional
        Filter by modality (e.g., "mri", "eeg").
    min_subjects : int, optional
        Minimum number of subjects.
    max_subjects : int, optional
        Maximum number of subjects.
    min_downloads : int, optional
        Minimum download count.
    has_readme : bool
        Only include datasets with readme.
    limit : int
        Maximum results (default: 20).
    order_by : str
        Order by: downloads, views, n_subjects, size_gb, name.

    Returns
    -------
    list
        List of matching datasets.
    """
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


@mcp.tool
def dataset_db_stats() -> Dict[str, Any]:
    """Get local dataset database statistics - use whenever the user asks how many datasets are indexed, counts per source (OpenNeuro/DANDI/PhysioNet/Zenodo), database size on disk, last build time, or the health/freshness of the scitex-dataset local index. Drop-in replacement for manual SQLite COUNT queries against the cache database.

    Returns
    -------
    dict
        Statistics including counts per source, last build time, etc.
    """
    from .. import database

    return database.get_stats()


# §5 — skills introspection tools (per audit-mcp-tools convention)
@mcp.tool
def dataset_skills_list() -> str:
    """List the names of every skill page shipped by scitex-dataset.

    Returns
    -------
        JSON string with `{"success": true, "package": "scitex-dataset",
        "skills": ["01_quick-start", "02_data-sources", ...]}`.
    """
    import json
    from pathlib import Path

    try:
        skills_dir = Path(__file__).parent.parent / "_skills" / "scitex-dataset"
        names = sorted(p.stem for p in skills_dir.glob("*.md") if p.name != "SKILL.md")
        return json.dumps(
            {"success": True, "package": "scitex-dataset", "skills": names},
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool
def dataset_skills_get(name: str) -> str:
    """Fetch the full Markdown content of one scitex-dataset skill page.

    Args:
        name: Skill page name without `.md`, e.g. `01_quick-start`.

    Returns
    -------
        JSON string with `{"success": true, "package": "scitex-dataset",
        "name": <name>, "content": <markdown>}`, or an error envelope.
    """
    import json
    from pathlib import Path

    try:
        skills_dir = Path(__file__).parent.parent / "_skills" / "scitex-dataset"
        target = skills_dir / f"{name}.md"
        if not target.exists():
            available = sorted(
                p.stem for p in skills_dir.glob("*.md") if p.name != "SKILL.md"
            )
            return json.dumps(
                {
                    "success": False,
                    "error": f"unknown skill {name!r}; available: {available}",
                },
                indent=2,
            )
        return json.dumps(
            {
                "success": True,
                "package": "scitex-dataset",
                "name": name,
                "content": target.read_text(encoding="utf-8"),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.resource("scitex-dataset://readme")
def get_readme() -> str:
    """Get package README."""
    from pathlib import Path

    readme_path = Path(__file__).parent.parent.parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    return "scitex-dataset: Unified interface for scientific dataset discovery."


if __name__ == "__main__":
    mcp.run()

# EOF
