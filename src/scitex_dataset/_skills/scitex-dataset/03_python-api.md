---
description: |
  [TOPIC] Python API
  [DETAILS] Public callables — search_datasets, sort_datasets, fetch_datasets, fetch_all_datasets, format_dataset, plus per-domain submodules.
tags: [scitex-dataset-python-api]
---

# Python API

```python
import scitex_dataset
```

## Top-level exports (`__all__`)

| Symbol | Purpose |
|---|---|
| `search_datasets(query)` | Search across all configured sources |
| `sort_datasets(results, by=...)` | Sort by `downloads`, `date`, `title`, … |
| `fetch_datasets(query=...)` | Fetch from OpenNeuro (default convenience) |
| `fetch_all_datasets()` | Pull every source's index, cached locally |
| `format_dataset(ds)` | Format a dataset record for display |
| `OPENNEURO_API` | Endpoint constant |
| `__version__` | Package version string |

## Domain submodules

| Submodule | Sources |
|---|---|
| `scitex_dataset.neuroscience` | OpenNeuro, DANDI, PhysioNet |
| `scitex_dataset.biology` | GEO |
| `scitex_dataset.medical` | ClinicalTrials.gov |
| `scitex_dataset.pharmacology` | ChEMBL |
| `scitex_dataset.general` | Zenodo, Scientific Data |
| `scitex_dataset.database` | Local index build + search |

Each submodule exposes `fetch(...)`, `search(...)`, and source-specific
helpers — see `dir(scitex_dataset.neuroscience.openneuro)` etc.

## Examples

```python
from scitex_dataset import search_datasets, sort_datasets
from scitex_dataset.neuroscience import openneuro, dandi
from scitex_dataset.database import build, search as db_search

results = search_datasets("Alzheimer EEG")
top = sort_datasets(results, by="downloads")[:10]

ds = openneuro.fetch("ds003104")
db_search("Neuropixels")
```

## See also

- [14_data-sources.md](14_data-sources.md) — per-source endpoint details
- [11_mcp-tools.md](11_mcp-tools.md) — MCP equivalents
