#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/pharmacology/chembl.py

"""
ChEMBL dataset fetcher.

ChEMBL hosts bioactivity data for drug-like molecules from EMBL-EBI.

API: https://www.ebi.ac.uk/chembl/api/data/

Example:
    >>> from scitex_dataset.pharmacology import chembl
    >>> datasets = chembl.fetch_all_datasets(max_datasets=10)
    >>> formatted = [chembl.format_dataset(ds) for ds in datasets]
"""

from __future__ import annotations

from typing import Optional

import httpx as _httpx
from scitex_dev.decorators import supports_return_as

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

__all__ = [
    "CHEMBL_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]


@supports_return_as
def fetch_datasets(
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Fetch a single page of assays from ChEMBL."""
    response = _httpx.get(
        f"{CHEMBL_API}/assay.json",
        params={
            "limit": limit,
            "offset": offset,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """Fetch all assays from ChEMBL with pagination."""
    all_datasets = []
    offset = 0
    limit = 100

    while True:
        try:
            result = fetch_datasets(limit=limit, offset=offset)
        except _httpx.HTTPStatusError as exc:
            if logger:
                logger.error(f"HTTP Error: {exc}")
            break
        except _httpx.RequestError as exc:
            if logger:
                logger.error(f"Request Error: {exc}")
            break

        assays = result.get("assays", [])
        meta = result.get("page_meta", {})
        has_next = bool(meta.get("next"))

        if not assays:
            break

        all_datasets.extend(assays)

        if logger:
            logger.info(f"Fetched {len(all_datasets)} ChEMBL assays...")

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        if not has_next:
            break

        offset += limit

    return all_datasets


@supports_return_as
def format_dataset(dataset: dict) -> dict:
    """Extract and format ChEMBL assay information."""
    chembl_id = dataset.get("assay_chembl_id", "")
    description = dataset.get("description", "N/A")
    assay_type = dataset.get("assay_type", "")
    organism = dataset.get("assay_organism", "")
    target = dataset.get("target_chembl_id", "")

    return {
        "id": chembl_id,
        "name": description,
        "assay_type": assay_type,
        "organism": organism,
        "target": target,
        "created": "",  # ChEMBL API does not expose creation date
        "url": f"https://www.ebi.ac.uk/chembl/assay_report_card/{chembl_id}/"
        if chembl_id
        else "",
        # ChEMBL-specific
        "assay_category": dataset.get("assay_category", ""),
        "assay_cell_type": dataset.get("assay_cell_type", ""),
    }


# EOF
