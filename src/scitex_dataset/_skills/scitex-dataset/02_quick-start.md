---
description: |
  [TOPIC] Quick start
  [DETAILS] Smallest example — search across all sources, fetch from OpenNeuro, sort by downloads.
tags: [scitex-dataset-quick-start]
---

# Quick Start

## Search all sources

```python
from scitex_dataset import search_datasets

results = search_datasets("EEG epilepsy")
for ds in results:
    print(f"{ds.id}: {ds.title}")
```

## Fetch from a specific source

```python
from scitex_dataset import fetch_datasets, fetch_all_datasets

eeg = fetch_datasets(query="resting state EEG")     # OpenNeuro
all_ds = fetch_all_datasets()                       # cached locally
```

## Sort

```python
from scitex_dataset import sort_datasets
top = sort_datasets(results, by="downloads")
```

## CLI

```bash
scitex-dataset --help
scitex-dataset neuroscience openneuro fetch -n 10
```

## Domains

- `scitex_dataset.neuroscience` — OpenNeuro, DANDI, PhysioNet
- `scitex_dataset.biology` — GEO
- `scitex_dataset.medical` — ClinicalTrials.gov
- `scitex_dataset.pharmacology` — ChEMBL
- `scitex_dataset.general` — Zenodo, Scientific Data
- `scitex_dataset.database` — local DB build + search

## Next

- [03_python-api.md](03_python-api.md) — full surface
- [04_cli-reference.md](04_cli-reference.md) — CLI commands + flags
- [13_quick-start.md](13_quick-start.md), [14_data-sources.md](14_data-sources.md) — historical notes
