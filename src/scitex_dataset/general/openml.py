#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenML API client for machine learning dataset discovery.

OpenML is an open platform for sharing machine learning datasets,
tasks, and experiments. It hosts thousands of curated ML datasets.

API Documentation: https://www.openml.org/apis
"""

from typing import Optional

import httpx
from scitex_dev.decorators import supports_return_as

OPENML_API = "https://www.openml.org/api/v1/json"


@supports_return_as
def fetch_datasets(
    offset: int = 0,
    limit: int = 100,
    status: str = "active",
) -> dict:
    """
    Fetch datasets from OpenML.

    Parameters
    ----------
    offset : int
        Offset for pagination.
    limit : int
        Number of results per request (max 10000).
    status : str
        Dataset status filter: 'active', 'deactivated', 'all'.

    Returns
    -------
    dict
        API response with dataset list.
    """
    params = {
        "offset": offset,
        "limit": limit,
        "status": status,
    }
    response = httpx.get(
        f"{OPENML_API}/data/list",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    page_size: int = 100,
    logger=None,
) -> list[dict]:
    """
    Fetch all datasets from OpenML with pagination.

    Parameters
    ----------
    max_datasets : int, optional
        Maximum number of datasets to fetch.
    page_size : int
        Datasets per request.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    list[dict]
        List of raw dataset dictionaries.
    """
    all_datasets = []
    offset = 0

    while True:
        if logger:
            logger.info(f"Fetching OpenML offset {offset}...")

        try:
            data = fetch_datasets(offset=offset, limit=page_size)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 412:
                # 412 = no results at this offset
                break
            raise

        datasets = data.get("data", {}).get("dataset", [])
        if not datasets:
            break

        all_datasets.extend(datasets)

        if logger:
            logger.info(
                f"  Retrieved {len(datasets)} records (total: {len(all_datasets)})"
            )

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        if len(datasets) < page_size:
            break

        offset += page_size

    return all_datasets


@supports_return_as
def format_dataset(record: dict) -> dict:
    """
    Format an OpenML dataset into a standardized dataset dictionary.

    Parameters
    ----------
    record : dict
        Raw OpenML dataset from list API.

    Returns
    -------
    dict
        Standardized dataset dictionary.
    """
    did = record.get("did", "")
    name = record.get("name", "")
    n_instances = record.get("NumberOfInstances", 0)
    n_features = record.get("NumberOfFeatures", 0)
    n_classes = record.get("NumberOfClasses", 0)
    n_missing = record.get("NumberOfMissingValues", 0)

    # Qualities that OpenML provides
    qualities = record.get("quality", [])
    quality_dict = {}
    if isinstance(qualities, list):
        for q in qualities:
            if isinstance(q, dict):
                quality_dict[q.get("name", "")] = q.get("value", "")

    return {
        "id": str(did),
        "doi": "",
        "name": name,
        "description": "",
        "created": record.get("upload_date", ""),
        "modified": record.get("upload_date", ""),
        "version": str(record.get("version", "")),
        "authors": [],
        "keywords": [],
        "license": "",
        "n_instances": int(n_instances) if n_instances else 0,
        "n_features": int(n_features) if n_features else 0,
        "n_classes": int(n_classes) if n_classes else 0,
        "n_missing_values": int(n_missing) if n_missing else 0,
        "n_downloads": int(record.get("NumberOfDownloads", 0) or 0),
        "views": 0,
        "downloads": int(record.get("NumberOfDownloads", 0) or 0),
        "url": f"https://www.openml.org/d/{did}",
        "source": "openml",
    }


# EOF
