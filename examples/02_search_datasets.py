#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 02: Search and filter datasets.

Demonstrates `search_datasets` and `sort_datasets` with several criteria,
writing the filtered output to the @stx.session output directory.

Usage:
    python 02_search_datasets.py
"""

import json
from pathlib import Path

import scitex as stx

from scitex_dataset import (
    fetch_all_datasets,
    format_dataset,
    search_datasets,
    sort_datasets,
)


@stx.session
def main(
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Fetch, search, sort, and write a filtered dataset list as JSON."""
    OUT = Path(CONFIG.SDIR_OUT)

    logger.info("Fetching datasets from OpenNeuro...")
    raw_datasets = fetch_all_datasets(max_datasets=100)
    datasets = [format_dataset(d) for d in raw_datasets]
    logger.info(f"Total: {len(datasets)} datasets")

    mri_datasets = search_datasets(datasets, modality="mri")
    logger.info(f"MRI datasets: {len(mri_datasets)}")

    large_mri = search_datasets(
        datasets,
        modality="mri",
        min_subjects=20,
        has_readme=True,
    )
    logger.info(f"MRI with 20+ subjects and readme: {len(large_mri)}")

    memory_datasets = search_datasets(datasets, text_query="memory")
    logger.info(f"Datasets mentioning 'memory': {len(memory_datasets)}")

    popular = sort_datasets(datasets, by="downloads", descending=True)[:10]
    logger.info("Top 10 most downloaded:")
    for ds in popular:
        logger.info(f"  {ds['id']}: {ds['downloads']} downloads - {ds['name']}")

    output_path = OUT / "mri_large_datasets.json"
    with open(output_path, "w") as f:
        json.dump(large_mri, f, indent=2)
    logger.info(f"Saved large MRI datasets to {output_path}")
    return 0


if __name__ == "__main__":
    main()

# EOF
