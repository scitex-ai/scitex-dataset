#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/neuroscience/physionet.py

"""
PhysioNet dataset fetcher.

PhysioNet hosts physiological signal databases including EEG, ECG, EMG,
and other biomedical signals.

API: https://physionet.org/api/v1/

Example:
    >>> from scitex_dataset.neuroscience import physionet
    >>> datasets = physionet.fetch_all_datasets(max_datasets=10)
    >>> formatted = [physionet.format_dataset(ds) for ds in datasets]
"""

from __future__ import annotations

from typing import Optional

import httpx as _httpx
from scitex_dev.decorators import supports_return_as

PHYSIONET_API = "https://physionet.org"

__all__ = [
    "PHYSIONET_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]


#: The published-projects listing. ``/rest/database-list/`` (the old
#: endpoint) answers 404 since PhysioNet's API move; this one returns every
#: published project version, unpaginated.
PUBLISHED_ENDPOINT = "/api/v1/project/published/"


@supports_return_as
def fetch_datasets(page: int = 1) -> list:
    """Fetch the published PhysioNet projects (LATEST version of each).

    The endpoint is not paginated; ``page`` > 1 returns an empty list so the
    pagination loop of :func:`fetch_all_datasets` ends after one call.
    """
    if page > 1:
        return []
    response = _httpx.get(f"{PHYSIONET_API}{PUBLISHED_ENDPOINT}", timeout=60.0)
    response.raise_for_status()
    return [p for p in response.json() if p.get("is_latest_version", True)]


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """Fetch all databases from PhysioNet with pagination."""
    all_datasets = []
    page = 1

    while True:
        try:
            result = fetch_datasets(page=page)
        except _httpx.HTTPStatusError as exc:
            if logger:
                logger.error(f"HTTP Error: {exc}")
            break
        except _httpx.RequestError as exc:
            if logger:
                logger.error(f"Request Error: {exc}")
            break

        # PhysioNet returns a list directly or paginated dict
        if isinstance(result, list):
            databases = result
            has_next = False
        else:
            databases = result.get("results", result.get("databases", []))
            has_next = bool(result.get("next"))

        if not databases:
            break

        all_datasets.extend(databases)

        if logger:
            logger.info(f"Fetched {len(all_datasets)} databases...")

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        if not has_next:
            break

        page += 1

    return all_datasets


@supports_return_as
def format_dataset(database: dict) -> dict:
    """Extract and format PhysioNet database information."""
    # Handle different response formats
    slug = database.get("slug", database.get("short_name", ""))
    title = database.get("title", database.get("name", "N/A"))

    size = database.get("main_storage_size") or database.get("total_size") or 0
    topics = [str(t).lower() for t in database.get("topics") or [] if t]
    version = database.get("version", "")
    return {
        "id": slug,
        "name": title,
        "version": version,
        "description": database.get("short_description", ""),
        "abstract": database.get("abstract", database.get("description", "")),
        "readme": _plain_text(database.get("abstract", "")),
        # The VERSION DOI pins exactly what a project uses; the core DOI
        # names the project across versions.
        "doi": database.get("version_doi")
        or database.get("core_doi")
        or database.get("doi", ""),
        "core_doi": database.get("core_doi", ""),
        "license": database.get("license", {}).get("name", "")
        if isinstance(database.get("license"), dict)
        else database.get("license", ""),
        "n_subjects": database.get("subject_count", 0),
        "n_records": database.get("record_count", 0),
        "size_bytes": int(size),
        "size_gb": round(size / (1024**3), 2),
        "publish_date": database.get("publish_date", ""),
        "modified": database.get("publish_date", ""),
        "modalities": topics,
        "tasks": topics,
        "url": database.get("source_url")
        or (f"https://physionet.org/content/{slug}/{version}/" if slug else ""),
        # PhysioNet-specific: Open / Restricted / Credentialed access.
        "data_access": database.get("access_policy", database.get("data_access", "")),
    }


def _plain_text(html: str) -> str:
    """The abstract without markup, for full-text search."""
    import html as _html
    import re

    return _html.unescape(re.sub(r"<[^>]+>", " ", html or "")).strip()


# EOF
