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

### Fetch a known file with provenance (matches `dataset_fetch_file` / `dataset_fetch_reference` / `dataset_list_references`)

| Symbol | Purpose |
|---|---|
| `fetch_file(url, sha256, dest_dir, filename=None, *, timeout=60.0, replace=False, license=None, citation=None)` | Download one file, verify its sha256, keep it in `dest_dir`, return a `FetchRecord` |
| `fetch_reference(name, dest_dir, *, timeout=60.0, replace=False)` | Same, for a registered reference file (e.g. `"mnist"`), with its license + citation |
| `list_references()` | Registered reference files (`id`, `url`, `sha256`, `size_bytes`, `license`, `citation`, …) |
| `FetchRecord` | Frozen dataclass: `url`, `sha256`, `size_bytes`, `path`, `retrieved_at` (ISO UTC), `cached`, `license`, `citation`; `.to_dict()` |
| `FetchError` | Raised on a sha256 mismatch or truncated body |

Contract — no fallback, no silent skip:

- A file already in `dest_dir` with the expected sha256 is reused
  (`cached=True`, nothing downloaded).
- A file already there with a **different** sha256 raises `FetchError` and is
  left untouched, unless `replace=True`.
- A new download streams to a hidden `.part` file in `dest_dir`; it is renamed
  into place only if the byte count matches `Content-Length` and the sha256
  matches. Otherwise the partial file is deleted and `FetchError` is raised.

```python
from scitex_dataset import fetch_reference

rec = fetch_reference("mnist", "./data")
rec.path        # .../data/mnist.npz (x_train, y_train, x_test, y_test)
rec.sha256      # 731c5ac6...b673d1, verified
rec.to_dict()   # JSON-ready provenance (license, citation, retrieved_at, ...)
```

The registry lives in `scitex_dataset.general.reference.REFERENCES` (one
dict per file). `download_dataset("reference", "mnist", dest)` routes there
too.

### Database (matches `dataset_db_*` MCP tools)

| Symbol | Purpose |
|---|---|
| `db_build(sources=None)` | Build / refresh the full-text dataset index |
| `db_search(query, ...)` | Search the index (words AND; `"phrases"`; `or`; `-not`) |
| `db_show_stats()` | Index statistics — per-source counts, total, store, last build |

The index lives in the shared SciTeX store, so there is no path argument
and no path in the stats. `scitex_dataset.database.store_description()`
reports which store was resolved; `SCITEX_STORE_DSN` repoints it.

Query words are stemmed and English stopwords are dropped, which the
previous tokeniser did neither of: `studies` now finds a dataset that says
`study`, and a query made only of stopwords matches nothing rather than
everything.

## Domain submodules

| Submodule | Sources |
|---|---|
| `scitex_dataset.neuroscience` | OpenNeuro, DANDI, PhysioNet |
| `scitex_dataset.general` | Zenodo, Figshare, OpenML, HuggingFace, reference files |
| `scitex_dataset.biology` | GEO |
| `scitex_dataset.pharmacology` | MoleculeNet, ChEMBL |
| `scitex_dataset.medical` | ClinicalTrials.gov |
| `scitex_dataset.database` | Dataset-index build + search |

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

# 4) Build the dataset index and query it.
db_build()
db_search("Alzheimer EEG")
```

## See also

- [14_data-sources.md](14_data-sources.md) — per-source endpoint details
- [11_mcp-tools.md](11_mcp-tools.md) — MCP equivalents (1:1 with the aliases above)
- [04_cli-reference.md](04_cli-reference.md) — CLI grammar
