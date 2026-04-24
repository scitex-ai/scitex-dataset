#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-03-28 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/medical/clinicaltrials.py

"""
ClinicalTrials.gov dataset fetcher.

ClinicalTrials.gov hosts clinical study records from the U.S. National
Library of Medicine.

API: https://clinicaltrials.gov/api/v2/studies

Example:
    >>> from scitex_dataset.medical import clinicaltrials
    >>> datasets = clinicaltrials.fetch_all_datasets(max_datasets=10)
    >>> formatted = [clinicaltrials.format_dataset(ds) for ds in datasets]
"""

from __future__ import annotations

from typing import Optional

import httpx as _httpx
from scitex_dev.decorators import supports_return_as

CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2"

__all__ = [
    "CLINICALTRIALS_API",
    "fetch_datasets",
    "fetch_all_datasets",
    "format_dataset",
]


@supports_return_as
def fetch_datasets(
    page_size: int = 100,
    page_token: Optional[str] = None,
) -> dict:
    """Fetch a single page of studies from ClinicalTrials.gov."""
    params = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    response = _httpx.get(
        f"{CLINICALTRIALS_API}/studies",
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """Fetch all studies from ClinicalTrials.gov with pagination."""
    all_datasets = []
    page_token = None

    while True:
        try:
            result = fetch_datasets(page_token=page_token)
        except _httpx.HTTPStatusError as exc:
            if logger:
                logger.error(f"HTTP Error: {exc}")
            break
        except _httpx.RequestError as exc:
            if logger:
                logger.error(f"Request Error: {exc}")
            break

        studies = result.get("studies", [])
        next_page_token = result.get("nextPageToken")

        if not studies:
            break

        all_datasets.extend(studies)

        if logger:
            logger.info(f"Fetched {len(all_datasets)} ClinicalTrials studies...")

        if max_datasets and len(all_datasets) >= max_datasets:
            all_datasets = all_datasets[:max_datasets]
            break

        if not next_page_token:
            break

        page_token = next_page_token

    return all_datasets


@supports_return_as
def format_dataset(dataset: dict) -> dict:
    """Extract and format ClinicalTrials.gov study information."""
    protocol = dataset.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    design_module = protocol.get("designModule", {})
    conditions_module = protocol.get("conditionsModule", {})

    nct_id = id_module.get("nctId", "")
    title = id_module.get("briefTitle", "N/A")

    # Enrollment info
    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment_count = enrollment_info.get("count", 0)

    # Phase
    phases = design_module.get("phases", [])
    phase = ", ".join(phases) if phases else ""

    # Conditions
    conditions = conditions_module.get("conditions", [])
    condition = ", ".join(conditions[:3]) if conditions else ""

    # Status
    overall_status = status_module.get("overallStatus", "")

    # Start date
    start_date_struct = status_module.get("startDateStruct", {})
    start_date = start_date_struct.get("date", "")

    return {
        "id": nct_id,
        "name": title,
        "n_subjects": int(enrollment_count) if enrollment_count else 0,
        "phase": phase,
        "condition": condition,
        "status": overall_status,
        "created": start_date,
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        # ClinicalTrials-specific
        "study_type": design_module.get("studyType", ""),
    }


# EOF
