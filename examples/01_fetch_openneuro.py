#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 01: Fetch datasets from OpenNeuro.

Fetches dataset metadata from OpenNeuro and writes a JSON snapshot
to the @stx.session output directory.

Usage:
    python 01_fetch_openneuro.py
"""

import json
from pathlib import Path

import scitex as stx

from scitex_dataset import fetch_all_datasets, format_dataset


@stx.session
def main(
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Fetch and format OpenNeuro datasets, then save as JSON."""
    OUT = Path(CONFIG.SDIR_OUT)

    logger.info("Fetching datasets from OpenNeuro...")
    raw_datasets = fetch_all_datasets(max_datasets=20)

    logger.info(f"Formatting {len(raw_datasets)} datasets...")
    datasets = [format_dataset(d) for d in raw_datasets]

    logger.info("First 5 datasets:")
    for ds in datasets[:5]:
        logger.info(f"  {ds['id']}: {ds['name']}")
        logger.info(
            f"    Subjects: {ds['n_subjects']}, Modality: {ds['primary_modality']}"
        )
        logger.info(f"    Downloads: {ds['downloads']}")

    output_path = OUT / "openneuro_datasets.json"
    with open(output_path, "w") as f:
        json.dump(datasets, f, indent=2)
    logger.info(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    main()

# EOF
