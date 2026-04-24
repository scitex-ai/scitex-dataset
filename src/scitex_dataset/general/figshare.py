#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figshare API client for scientific dataset discovery.

Figshare is a repository where users can make all of their research
outputs available in a citable, shareable and discoverable manner.

API Documentation: https://docs.figshare.com/
"""

from typing import Optional

import httpx
from scitex_dev.decorators import supports_return_as

FIGSHARE_API = "https://api.figshare.com/v2"


@supports_return_as
def fetch_datasets(
    query: str = "",
    page: int = 1,
    page_size: int = 25,
    order: str = "published_date",
    order_direction: str = "desc",
) -> list[dict]:
    """
    Fetch datasets from Figshare.

    Parameters
    ----------
    query : str
        Search query string.
    page : int
        Page number (1-indexed).
    page_size : int
        Number of results per page (max 1000).
    order : str
        Sort field: 'published_date', 'modified_date', 'views', 'shares'.
    order_direction : str
        Sort direction: 'asc' or 'desc'.

    Returns
    -------
    list[dict]
        List of article records.
    """
    params = {
        "page": page,
        "page_size": page_size,
        "order": order,
        "order_direction": order_direction,
        "item_type": 3,  # 3 = dataset
    }
    if query:
        params["search_for"] = query

    response = httpx.post(
        f"{FIGSHARE_API}/articles/search",
        json=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


@supports_return_as
def fetch_all_datasets(
    query: str = "",
    max_datasets: Optional[int] = None,
    page_size: int = 25,
    logger=None,
) -> list[dict]:
    """
    Fetch all datasets from Figshare with pagination.

    Parameters
    ----------
    query : str
        Search query string.
    max_datasets : int, optional
        Maximum number of datasets to fetch.
    page_size : int
        Datasets per request.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    list[dict]
        List of raw article dictionaries.
    """
    all_datasets = []
    page = 1

    while True:
        if logger:
            logger.info(f"Fetching Figshare page {page}...")

        hits = fetch_datasets(
            query=query,
            page=page,
            page_size=page_size,
        )

        if not hits:
            break

        all_datasets.extend(hits)

        if logger:
            logger.info(
                f"  Retrieved {len(hits)} records (total: {len(all_datasets)})"
            )

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        if len(hits) < page_size:
            break

        page += 1

    return all_datasets


@supports_return_as
def format_dataset(record: dict) -> dict:
    """
    Format a Figshare article into a standardized dataset dictionary.

    Parameters
    ----------
    record : dict
        Raw Figshare article from API.

    Returns
    -------
    dict
        Standardized dataset dictionary.
    """
    # Fetch full article details for richer metadata
    article_id = record.get("id", "")
    try:
        detail = httpx.get(
            f"{FIGSHARE_API}/articles/{article_id}",
            timeout=30,
        ).json()
    except Exception:
        detail = record

    authors = [a.get("full_name", "") for a in detail.get("authors", [])]
    files = detail.get("files", [])
    total_size = sum(f.get("size", 0) for f in files)
    tags = detail.get("tags", [])
    categories = [c.get("title", "") for c in detail.get("categories", [])]
    license_info = detail.get("license", {})
    license_name = license_info.get("name", "") if isinstance(license_info, dict) else str(license_info)

    return {
        "id": str(article_id),
        "doi": detail.get("doi", ""),
        "name": detail.get("title", record.get("title", "")),
        "description": detail.get("description", ""),
        "created": detail.get("created_date", ""),
        "modified": detail.get("modified_date", ""),
        "publish_date": detail.get("published_date", ""),
        "version": str(detail.get("version", "")),
        "authors": authors,
        "keywords": tags + categories,
        "license": license_name,
        "n_files": len(files),
        "size_bytes": total_size,
        "size_gb": round(total_size / (1024**3), 3) if total_size else 0,
        "views": detail.get("views", 0),
        "downloads": detail.get("downloads", 0),
        "url": detail.get("url_public_html", record.get("url", "")),
        "source": "figshare",
    }


# EOF
