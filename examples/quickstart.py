#!/usr/bin/env python3
"""scitex-dataset quickstart (offline-safe).

Imports the package and demonstrates non-network introspection.
Real data fetching is shown in 01_fetch_openneuro.py / 02_search_datasets.py.

Usage:
    python quickstart.py
"""

from __future__ import annotations

import scitex as stx

import scitex_dataset as sxd


@stx.session
def main(
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Smoke-test scitex-dataset import + offline filtering."""
    _ = CONFIG  # output dir not needed for offline smoke test
    logger.info(
        f"scitex-dataset modules: {[m for m in dir(sxd) if not m.startswith('_')]}"
    )
    logger.info(f"OpenNeuro API endpoint: {sxd.OPENNEURO_API}")

    # search_datasets is pure-python over local list — exercise without network
    sample = [
        {"name": "ds-001", "modality": "mri", "subjects": 10},
        {"name": "ds-002", "modality": "eeg", "subjects": 5},
    ]
    mri = sxd.search_datasets(sample, modality="mri")
    logger.info(f"Filtered MRI sample: {mri}")

    logger.info("scitex-dataset import + offline API OK")
    return 0


if __name__ == "__main__":
    main()
