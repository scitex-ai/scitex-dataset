---
name: scitex-dataset
description: |
  [WHAT] Unified dataset-discovery API across 11 scientific repositories (10 catalog sources + HuggingFace on-demand).
  [WHEN] Use when the user asks to "find an EEG dataset", "list BIDS datasets on topic X", "search DANDI for Neuropixels", "get GEO series for Alzheimer's", "find a ChEMBL target / bioassay".
  [HOW] `import scitex_dataset` then call `fetch_all_datasets()`.
tags: [scitex-dataset]
allowed-tools: mcp__scitex__dataset_*
primary_interface: python
interfaces:
  python: 3
  cli: 1
  mcp: 2
  skills: 2
  hook: 0
  http: 0
---


# scitex-dataset

> **Interfaces:** Python ⭐⭐⭐ (primary) · CLI ⭐ · MCP ⭐⭐ · Skills ⭐⭐ · Hook — · HTTP —

Unified cross-repository dataset discovery — one API over OpenNeuro,
DANDI, PhysioNet, Zenodo, Figshare, OpenML, MoleculeNet, GEO, ChEMBL,
ClinicalTrials.gov (catalog-indexed) plus HuggingFace Hub (on-demand
fetch by repo_id).

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

### Mandatory
- [01_installation.md](01_installation.md) — pip install + extras + verify
- [02_quick-start.md](02_quick-start.md) — search / fetch / sort
- [03_python-api.md](03_python-api.md) — Public callables + domain submodules
- [04_cli-reference.md](04_cli-reference.md) — `scitex-dataset` console entry

### Workflows
- [10_cli-reference.md](10_cli-reference.md) — historical CLI notes
- [11_mcp-tools.md](11_mcp-tools.md) — MCP tools for AI agents
- [13_quick-start.md](13_quick-start.md) — historical quick-start
- [14_data-sources.md](14_data-sources.md) — OpenNeuro, DANDI, PhysioNet details

## CLI

```bash
scitex-dataset fetch_all_datasets "EEG epilepsy"
scitex-dataset fetch openneuro ds003104
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `dataset_list_sources` | Enumerate the 11 supported sources |
| `dataset_search` | Filter + rank an in-memory list of fetched datasets |
| `dataset_openneuro_fetch` | Fetch BIDS metadata from OpenNeuro |
| `dataset_dandi_fetch` | Fetch NWB metadata from DANDI Archive |
| `dataset_physionet_fetch` | Fetch PhysioNet databases |
| `dataset_zenodo_fetch` | Fetch Zenodo records (query) |
| `dataset_figshare_fetch` | Fetch Figshare items (query) |
| `dataset_openml_fetch` | Fetch OpenML datasets |
| `dataset_moleculenet_fetch` | Fetch MoleculeNet benchmarks |
| `dataset_geo_fetch` | Fetch GEO series |
| `dataset_chembl_fetch` | Fetch ChEMBL bioactivity records |
| `dataset_clinicaltrials_fetch` | Fetch ClinicalTrials.gov studies |
| `dataset_hf_fetch` | Snapshot-download a HuggingFace repo |
| `dataset_hf_search` | Search HuggingFace Hub |
| `dataset_hf_info` | Get HF dataset/model metadata |
| `dataset_hf_download_file` | Download one file from an HF repo |
| `dataset_db_build` | Build/refresh the local SQLite + FTS5 index |
| `dataset_db_search` | Offline search of the local index |
| `dataset_db_stats` | Local-index statistics |
| `dataset_skills_list` | List bundled skill pages |
| `dataset_skills_get` | Read a bundled skill page by name |


## Environment

- [12_env-vars.md](12_env-vars.md) — SCITEX_* env vars read by scitex-dataset at runtime
