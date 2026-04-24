#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/biology/geo.py

"""
GEO (Gene Expression Omnibus) dataset fetcher.

GEO hosts gene expression and genomics datasets from NCBI.

API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Example:
    >>> from scitex_dataset.biology import geo
    >>> datasets = geo.fetch_all_datasets(max_datasets=10)
    >>> formatted = [geo.format_dataset(ds) for ds in datasets]
"""

from __future__ import annotations

from typing import Optional

import httpx as _httpx
from scitex_dev.decorators import supports_return_as

GEO_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

__all__ = [
    "GEO_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]


@supports_return_as
def fetch_datasets(
    retstart: int = 0,
    retmax: int = 100,
    term: str = "gds[Entry Type]",
) -> dict:
    """Fetch a single page of datasets from GEO via NCBI E-utilities."""
    # Step 1: Search for GDS IDs
    search_response = _httpx.get(
        f"{GEO_API}/esearch.fcgi",
        params={
            "db": "gds",
            "term": term,
            "retstart": retstart,
            "retmax": retmax,
            "retmode": "json",
        },
        timeout=30.0,
    )
    search_response.raise_for_status()
    search_data = search_response.json()

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    total_count = int(search_data.get("esearchresult", {}).get("count", 0))

    if not id_list:
        return {"results": [], "total": total_count}

    # Step 2: Fetch summaries for those IDs
    summary_response = _httpx.get(
        f"{GEO_API}/esummary.fcgi",
        params={
            "db": "gds",
            "id": ",".join(id_list),
            "retmode": "json",
        },
        timeout=30.0,
    )
    summary_response.raise_for_status()
    summary_data = summary_response.json()

    results = []
    result_block = summary_data.get("result", {})
    for uid in result_block.get("uids", []):
        entry = result_block.get(uid)
        if entry:
            results.append(entry)

    return {"results": results, "total": total_count}


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """Fetch all datasets from GEO with pagination."""
    all_datasets = []
    retstart = 0
    retmax = 100

    while True:
        try:
            result = fetch_datasets(retstart=retstart, retmax=retmax)
        except _httpx.HTTPStatusError as exc:
            if logger:
                logger.error(f"HTTP Error: {exc}")
            break
        except _httpx.RequestError as exc:
            if logger:
                logger.error(f"Request Error: {exc}")
            break

        datasets = result.get("results", [])
        total = result.get("total", 0)

        if not datasets:
            break

        all_datasets.extend(datasets)

        if logger:
            logger.info(f"Fetched {len(all_datasets)}/{total} GEO datasets...")

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        retstart += retmax
        if retstart >= total:
            break

    return all_datasets


@supports_return_as
def format_dataset(dataset: dict) -> dict:
    """Extract and format GEO dataset information."""
    uid = str(dataset.get("uid", ""))
    accession = dataset.get("accession", dataset.get("Accession", ""))
    title = dataset.get("title", dataset.get("Title", "N/A"))
    n_samples = dataset.get("n_samples", dataset.get("NSamples", 0))
    organism = dataset.get("taxon", dataset.get("Organism", ""))
    platform = dataset.get("gpl", dataset.get("GPL", ""))
    pdat = dataset.get("pdat", dataset.get("PDAT", ""))

    dataset_id = accession or uid

    return {
        "id": dataset_id,
        "name": title,
        "n_subjects": int(n_samples) if n_samples else 0,
        "organism": organism,
        "platform": platform,
        "created": pdat,
        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset_id}"
        if dataset_id
        else "",
        # GEO-specific
        "gds_type": dataset.get("gdstype", dataset.get("GDSType", "")),
    }


# EOF
