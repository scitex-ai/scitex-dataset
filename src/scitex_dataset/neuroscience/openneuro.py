#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:35:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/neuroscience/openneuro.py

"""
OpenNeuro dataset fetcher using GraphQL API.

Example:
    >>> from scitex_dataset import fetch_all_datasets, format_dataset
    >>> datasets = fetch_all_datasets(max_datasets=10)
    >>> formatted = [format_dataset(ds) for ds in datasets]
"""

from __future__ import annotations

from typing import Optional

import httpx as _httpx  # noqa: N812
from scitex_dev.decorators import supports_return_as

OPENNEURO_API = "https://openneuro.org/crn/graphql"

__all__ = [
    "OPENNEURO_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]


def _make_query(first: int = 10, after: Optional[str] = None) -> str:
    after_arg = f', after: "{after}"' if after else ""
    return f"""
query {{
  datasets(first: {first}{after_arg}) {{
    edges {{
      node {{
        id
        name
        created
        public
        publishDate
        analytics {{ views downloads }}
        draft {{
          modified
          readme
          description {{
            Name BIDSVersion License Authors SeniorAuthor
            DatasetDOI DatasetType Acknowledgements
            HowToAcknowledge Funding ReferencesAndLinks EthicsApprovals
          }}
          summary {{
            modalities primaryModality secondaryModalities
            sessions subjects tasks size totalFiles dataProcessed
          }}
        }}
      }}
    }}
    pageInfo {{ hasNextPage endCursor }}
  }}
}}
"""


@supports_return_as
def fetch_datasets(first: int = 10, after: Optional[str] = None) -> dict:
    """Fetch a single page of datasets from OpenNeuro."""
    response = _httpx.post(
        OPENNEURO_API,
        json={"query": _make_query(first, after)},
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


@supports_return_as
def fetch_all_datasets(
    batch_size: int = 100,
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """Fetch every dataset record from OpenNeuro by paginating GraphQL.

    Walks the public ``crn/graphql`` endpoint with cursor-based
    pagination until exhausted (or ``max_datasets`` is reached). Use
    ``format_dataset`` to project each raw record into the package's
    common dataset schema.

    Parameters
    ----------
    batch_size : int, default 100
        Records per HTTP request. The OpenNeuro server caps this; the
        function does not validate the upper bound.
    max_datasets : int, optional
        Stop after this many records. ``None`` (default) fetches the
        entire catalog.
    logger : logging.Logger, optional
        If provided, HTTP and GraphQL errors are logged. Errors are
        otherwise silent (the function returns whatever it has so far).

    Returns
    -------
    list[dict]
        Raw GraphQL ``node`` dicts, in catalog order. Pass each through
        ``format_dataset`` for the normalized schema.

    Examples
    --------
    >>> records = fetch_all_datasets(max_datasets=10)
    >>> len(records) <= 10
    True
    """
    all_datasets = []
    cursor = None

    while True:
        try:
            result = fetch_datasets(first=batch_size, after=cursor)
        except _httpx.HTTPStatusError as exc:
            if logger:
                logger.error(f"HTTP Error: {exc}")
            break
        except _httpx.RequestError as exc:
            if logger:
                logger.error(f"Request Error: {exc}")
            break

        if "errors" in result:
            if logger:
                logger.error(f"GraphQL Errors: {result['errors']}")
            break

        datasets = result.get("data", {}).get("datasets", {})
        edges = datasets.get("edges", [])
        page_info = datasets.get("pageInfo", {})

        for edge in edges:
            all_datasets.append(edge["node"])

        if logger:
            logger.info(f"Fetched {len(all_datasets)} datasets...")

        if max_datasets and len(all_datasets) >= max_datasets:
            break

        if not page_info.get("hasNextPage"):
            break

        cursor = page_info.get("endCursor")

    return all_datasets


@supports_return_as
def format_dataset(node: dict) -> dict:
    """Project a raw OpenNeuro GraphQL node into the common dataset schema.

    Every catalog source exposes ``format_dataset`` returning the same
    shape so they can plug into ``database.build`` and
    ``search.search_datasets`` uniformly.

    Parameters
    ----------
    node : dict
        A single ``edges[].node`` element from the OpenNeuro GraphQL
        response (the ``draft`` / ``analytics`` keys are read; missing
        fields fall back to ``None`` / 0).

    Returns
    -------
    dict
        Normalized record with keys: ``id, name, n_subjects,
        modalities, tasks, size_gb, downloads, views, readme,
        license, doi, url, source``.
    """
    draft = node.get("draft") or {}
    description = draft.get("description") or {}
    summary = draft.get("summary") or {}
    analytics = node.get("analytics") or {}

    size_bytes = summary.get("size") or 0
    size_gb = size_bytes / (1024**3)

    return {
        "id": node["id"],
        "name": node.get("name") or description.get("Name", "N/A"),
        "created": node.get("created"),
        "modified": draft.get("modified"),
        "publish_date": node.get("publishDate"),
        "public": node.get("public"),
        "views": analytics.get("views"),
        "downloads": analytics.get("downloads"),
        "readme": draft.get("readme"),
        "bids_version": description.get("BIDSVersion"),
        "license": description.get("License"),
        "authors": description.get("Authors"),
        "senior_author": description.get("SeniorAuthor"),
        "doi": description.get("DatasetDOI"),
        "dataset_type": description.get("DatasetType"),
        "acknowledgements": description.get("Acknowledgements"),
        "how_to_acknowledge": description.get("HowToAcknowledge"),
        "funding": description.get("Funding"),
        "references_and_links": description.get("ReferencesAndLinks"),
        "ethics_approvals": description.get("EthicsApprovals"),
        "modalities": summary.get("modalities", []),
        "primary_modality": summary.get("primaryModality"),
        "secondary_modalities": summary.get("secondaryModalities", []),
        "sessions": summary.get("sessions", []),
        "n_subjects": len(summary.get("subjects") or []),
        "tasks": summary.get("tasks", []),
        "size_gb": round(size_gb, 2),
        "total_files": summary.get("totalFiles", 0),
        "data_processed": summary.get("dataProcessed"),
    }


def _log_dataset(dataset: dict, logger) -> None:
    """Log formatted dataset information (internal use)."""
    logger.info(f"ID: {dataset['id']}")
    logger.info(f"  Name: {dataset['name']}")
    logger.info(f"  Modalities: {dataset['modalities']}")
    logger.info(f"  Subjects: {dataset['n_subjects']}")
    logger.info(f"  Size: {dataset['size_gb']:.2f} GB")


# EOF
