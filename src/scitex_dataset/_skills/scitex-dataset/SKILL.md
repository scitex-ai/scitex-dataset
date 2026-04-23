---
description: Dataset fetcher for neuroscience research — OpenNeuro, DANDI, PhysioNet with local database fetch_all_datasets and BIDS support.
allowed-tools: mcp__scitex__dataset_*
---

# scitex-dataset

Dataset fetch_all_datasets and fetch for neuroscience research.

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
