#!/usr/bin/env python3
"""scitex-dataset quickstart (offline-safe).

Imports the package and demonstrates non-network introspection.
Real data fetching is shown in 01_fetch_openneuro.py / 02_search_datasets.py.
"""

from __future__ import annotations

import scitex_dataset as sxd


def main() -> int:
    print(f"scitex-dataset modules: {[m for m in dir(sxd) if not m.startswith('_')]}")
    print(f"OpenNeuro API endpoint: {sxd.OPENNEURO_API}")

    # search_datasets is pure-python over local list — exercise without network
    sample = [
        {"name": "ds-001", "modality": "mri", "subjects": 10},
        {"name": "ds-002", "modality": "eeg", "subjects": 5},
    ]
    mri = sxd.search_datasets(sample, modality="mri")
    print(f"Filtered MRI sample: {mri}")

    print("scitex-dataset import + offline API OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
