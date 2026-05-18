---
description: |
  [TOPIC] Python API
  [DETAILS] Top-level exports (search/sort/fetch + per-source aliases),
  the domain submodules, and the database / filter / list_sources
  helpers that mirror the MCP tool surface.
tags: [scitex-dataset-python-api]
---

# Python API

```python
import scitex_dataset
```

## Top-level exports (`__all__`)

### Convenience (OpenNeuro defaults)

| Symbol | Purpose |
|---|---|
| `fetch_datasets(query=...)` | Fetch from OpenNeuro |
| `fetch_all_datasets()` | Iterate full OpenNeuro catalog |
| `format_dataset(ds)` | Normalize a record |
| `__version__` | Package version |

### Search + filter

| Symbol | Purpose |
|---|---|
| `search_datasets(records, **filters)` | Filter an in-memory list |
| `sort_datasets(records, by=...)` | Sort by `downloads`, `date`, … |
| `filter_results(records, **filters, sort_by=...)` | Search + sort + slice (matches the ``dataset_filter_results`` MCP tool) |
| `list_sources()` | Dict of the 11 supported sources (matches ``dataset_list_sources``) |

### Per-source `<src>_fetch` aliases

Every catalog source exposes a top-level ``<src>_fetch`` re-binding of
its ``fetch_all_datasets``, so the public API mirrors the MCP tool
naming (``dataset_<src>_fetch``):

```python
from scitex_dataset import (
    openneuro_fetch, dandi_fetch, physionet_fetch,
    zenodo_fetch, figshare_fetch, openml_fetch,
    moleculenet_fetch, geo_fetch, chembl_fetch, clinicaltrials_fetch,
)
```

### HuggingFace family

| Symbol | Purpose |
|---|---|
| `huggingface_fetch(query, ...)` | Catalog-style adapter (uses ``search_hub``) |
| `huggingface_search(query, ...)` | Live HuggingFace Hub search |
| `huggingface_info(repo_id, ...)` | Dataset / model metadata |
| `huggingface_download_file(...)` | Single-file download |

For ``snapshot_download``-style fetch by repo_id, use
``scitex_dataset.general.huggingface.fetch_dataset``.

### Database (matches `dataset_db_*` MCP tools)

| Symbol | Purpose |
|---|---|
| `db_build(sources=None)` | Build / refresh the local SQLite + FTS5 index |
| `db_search(query, ...)` | Offline search |
| `db_show_stats()` | Index statistics |

## Domain submodules

| Submodule | Sources |
|---|---|
| `scitex_dataset.neuroscience` | OpenNeuro, DANDI, PhysioNet |
| `scitex_dataset.general` | Zenodo, Figshare, OpenML, HuggingFace |
| `scitex_dataset.biology` | GEO |
| `scitex_dataset.pharmacology` | MoleculeNet, ChEMBL |
| `scitex_dataset.medical` | ClinicalTrials.gov |
| `scitex_dataset.database` | Local index build + search |

Each domain module exposes per-source modules with the standard
``fetch_all_datasets() / format_dataset(ds)`` contract.

## Examples

```python
from scitex_dataset import (
    openneuro_fetch, huggingface_search, filter_results, list_sources,
    db_build, db_search,
)

# 1) Fetch from any source via the alias.
records = openneuro_fetch(max_datasets=200)

# 2) Filter + rank in memory.
top = filter_results(records, modality="eeg", min_subjects=20, sort_by="downloads", limit=10)

# 3) Search HF Hub directly.
hf_hits = huggingface_search("biology", limit=20)

# 4) Build the local index and query offline.
db_build()
db_search("Alzheimer EEG")
```

## See also

- [14_data-sources.md](14_data-sources.md) — per-source endpoint details
- [11_mcp-tools.md](11_mcp-tools.md) — MCP equivalents (1:1 with the aliases above)
- [04_cli-reference.md](04_cli-reference.md) — CLI grammar
