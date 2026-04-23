---
description: Unified dataset-discovery API across 7 scientific repositories — OpenNeuro + DANDI + PhysioNet (neuroscience, BIDS + NWB), Zenodo + Scientific Data (general), GEO (gene expression), ChEMBL (pharmacology), ClinicalTrials.gov (medical). Public API — `fetch_all_datasets()` / `fetch_datasets()` / `format_dataset()` (OpenNeuro convenience) + `search_datasets()` / `sort_datasets()` (cross-source ranking) + domain submodules `neuroscience`, `general`, `biology`, `pharmacology`, `medical` + `database` (build & query a local unified SQLite index). 12 MCP tools — `dataset_search` (cross-source), per-source `dataset_openneuro_fetch` / `dataset_dandi_fetch` / `dataset_physionet_fetch` / `dataset_geo_fetch` / `dataset_zenodo_fetch` / `dataset_chembl_fetch` / `dataset_clinicaltrials_fetch`, and local-db `dataset_db_build` / `dataset_db_search` / `dataset_db_stats` / `dataset_list_sources`. Drop-in replacement for `openneuro-python`, `dandi` CLI, raw `requests` against the OpenNeuro GraphQL API / PhysioNet / GEO / ChEMBL / ClinicalTrials REST endpoints, `pyzenodo3`, and hand-rolled dataset-scrapers. Use whenever the user asks to "find an EEG dataset", "list BIDS datasets on topic X", "search DANDI for Neuropixels", "get GEO series for Alzheimer's", "find a ChEMBL target / bioassay", "search ClinicalTrials.gov", "index all datasets locally", "cross-search multiple dataset repositories", "sort datasets by subject count / modality", or mentions OpenNeuro, DANDI, PhysioNet, BIDS, NWB, GEO, ChEMBL, ClinicalTrials, Zenodo, Scientific Data.
allowed-tools: mcp__scitex__dataset_*
---

# scitex-dataset

Unified cross-repository dataset discovery — one API over OpenNeuro, DANDI, PhysioNet, Zenodo, Scientific Data, GEO, ChEMBL, and ClinicalTrials.gov.

## Installation & import (two equivalent paths)

The same module is reachable via two install paths. Both forms work at
runtime; which one a user has depends on their install choice.

```python
# Standalone — pip install scitex-dataset
import scitex_dataset
scitex_dataset.fetch_all_datasets(...)

# Umbrella — pip install scitex
import scitex.dataset
scitex.dataset.fetch_all_datasets(...)
```

`pip install scitex-dataset` alone does NOT expose the `scitex` namespace;
`import scitex.dataset` raises `ModuleNotFoundError`. To use the
`scitex.dataset` form, also `pip install scitex`.

See [../../general/02_interface-python-api.md] for the ecosystem-wide
rule and empirical verification table.

## Sub-skills

### Core
- [01_quick-start.md](01_quick-start.md) — Basic usage
- [02_data-sources.md](02_data-sources.md) — OpenNeuro, DANDI, PhysioNet

### Workflows
- [10_cli-reference.md](10_cli-reference.md) — CLI commands
- [11_mcp-tools.md](11_mcp-tools.md) — MCP tools for AI agents

## CLI

```bash
scitex-dataset fetch_all_datasets "EEG epilepsy"
scitex-dataset fetch openneuro ds003104
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `dataset_search` | Search across all sources |
| `dataset_openneuro_fetch` | Fetch from OpenNeuro |
| `dataset_dandi_fetch` | Fetch from DANDI Archive |
| `dataset_physionet_fetch` | Fetch from PhysioNet |
| `dataset_db_search` | Search local database |
| `dataset_db_build` | Build local database |
| `dataset_db_stats` | Database statistics |
| `dataset_list_sources` | List available sources |
