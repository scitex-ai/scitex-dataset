# SciTeX Dataset (<code>scitex-dataset</code>)

<!-- scitex-badges:start -->
[![PyPI](https://img.shields.io/pypi/v/scitex-dataset.svg)](https://pypi.org/project/scitex-dataset/)
[![Python](https://img.shields.io/pypi/pyversions/scitex-dataset.svg)](https://pypi.org/project/scitex-dataset/)
[![Tests](https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/test.yml)
[![Install Test](https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/install-test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/install-test.yml)
[![Coverage](https://codecov.io/gh/ywatanabe1989/scitex-dataset/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/scitex-dataset)
[![Docs](https://readthedocs.org/projects/scitex-dataset/badge/?version=latest)](https://scitex-dataset.readthedocs.io/en/latest/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- scitex-badges:end -->

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Unified access to neuroscience and scientific datasets</b></p>

<p align="center">
  <a href="https://scitex-dataset.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-dataset</code>
</p>

---

## Problem and Solution

| # | Problem | Solution |
|---|---------|----------|
| 1 | **Public dataset repositories balkanized** -- OpenNeuro (BIDS) + DANDI (NWB) + PhysioNet (WFDB) + Zenodo (generic) + GEO / ChEMBL / ClinicalTrials — different APIs, auth, download tools | **Unified fetcher** -- `stx.dataset.neuroscience.openneuro.fetch_all_datasets()` same call shape across all; local FTS5 search across metadata |
| 2 | **"Download this BIDS dataset" means reading DataLad docs first** -- the barrier is tooling, not knowledge | **One-line fetch** -- no DataLad setup; the module handles auth, resumption, checksums transparently |

## Supported repositories

| Repository | Description | Data Types |
|------------|-------------|------------|
| **OpenNeuro** | Open platform for sharing neuroimaging data | MRI, EEG, MEG, iEEG, PET |
| **DANDI** | BRAIN Initiative data archive | Electrophysiology, Ophys |
| **PhysioNet** | Physiological signal databases | ECG, EEG, clinical data |
| **Zenodo** | General scientific data repository (CERN) | Any research data |

<p align="center"><sub><b>Table 1.</b> Supported data repositories. Each source is queried via its public API; no authentication required for metadata access.</sub></p>

## Installation

Requires Python >= 3.10.

```bash
pip install scitex-dataset
```

> **MCP support**: `pip install scitex-dataset[mcp]`

## Quick Start

```python
from scitex_dataset import fetch_all_datasets, format_dataset

# Fetch datasets from OpenNeuro
datasets = fetch_all_datasets(max_datasets=10)

# Format for analysis
for ds in datasets:
    formatted = format_dataset(ds)
    print(f"{formatted['id']}: {formatted['name']} ({formatted['n_subjects']} subjects)")
```

## Four Interfaces

<details>
<summary><strong>Python API</strong></summary>

<br>

```python
from scitex_dataset import fetch_all_datasets, format_dataset, search_datasets, sort_datasets
from scitex_dataset import neuroscience, database

# Fetch from specific sources
datasets = fetch_all_datasets(max_datasets=100)                    # OpenNeuro
dandi_ds = neuroscience.dandi.fetch_all_datasets(max_datasets=50)  # DANDI
phys_ds = neuroscience.physionet.fetch_all_datasets()              # PhysioNet

# Search and filter
eeg_datasets = search_datasets(datasets, modality="eeg", min_subjects=20)
popular = sort_datasets(datasets, by="downloads", descending=True)

# Local database for fast full-text search
database.build()                                        # index all sources
results = database.search("alzheimer EEG", min_subjects=20)
```

> **[Full API reference](https://scitex-dataset.readthedocs.io/)**

</details>

<details>
<summary><strong>CLI Commands</strong></summary>

<br>

```bash
scitex-dataset --help-recursive             # Show all commands

# Fetch from repositories
scitex-dataset openneuro -n 100 -o datasets.json -v
scitex-dataset dandi -n 50 -o dandi.json -v
scitex-dataset physionet -n 50 -v
scitex-dataset zenodo -q "neuroscience" -n 20

# Local database
scitex-dataset db build                     # index all sources
scitex-dataset db search "epilepsy EEG"     # full-text search
scitex-dataset db stats                     # show statistics

# Introspection
scitex-dataset list-python-apis -v          # list Python API tree
scitex-dataset mcp list-tools -v            # list MCP tools
```

> **[Full CLI reference](https://scitex-dataset.readthedocs.io/)**

</details>

<details>
<summary><strong>MCP Server</strong></summary>

<br>

AI agents can discover and query neuroscience datasets autonomously.

| Tool | Description |
|------|-------------|
| `dataset_openneuro_fetch` | Fetch datasets from OpenNeuro |
| `dataset_dandi_fetch` | Fetch datasets from DANDI Archive |
| `dataset_physionet_fetch` | Fetch datasets from PhysioNet |
| `dataset_zenodo_fetch` | Fetch datasets from Zenodo |
| `dataset_search` | Filter datasets by modality, subjects, etc. |
| `dataset_list_sources` | List available data repositories |
| `dataset_db_build` | Build local search database |
| `dataset_db_search` | Full-text search across all sources |
| `dataset_db_stats` | Database statistics |

<sub><b>Table 2.</b> Nine MCP tools available for AI-assisted dataset discovery. All tools accept JSON parameters and return JSON results.</sub>

```bash
scitex-dataset mcp start
```

> **[Full MCP specification](https://scitex-dataset.readthedocs.io/)**

</details>

<details>
<summary><strong>Skills</strong></summary>

<br>

Skills provide workflow-oriented guides that AI agents query to discover capabilities and usage patterns.

```bash
scitex-dataset skills list              # List available skill pages
scitex-dataset skills get SKILL         # Show main skill page
scitex-dev skills export --package scitex-dataset  # Export to Claude Code
```

| Skill | Content |
|-------|---------|
| `quick-start` | Basic usage |
| `data-sources` | OpenNeuro, DANDI, PhysioNet |
| `cli-reference` | CLI commands |
| `mcp-tools` | MCP tools for AI agents |

</details>

## Part of SciTeX

SciTeX Dataset is part of [**SciTeX**](https://scitex.ai). When used inside the SciTeX framework, dataset discovery integrates with reproducible research sessions:

```python
import scitex
from scitex_dataset import fetch_all_datasets, format_dataset

@scitex.session
def main(logger=scitex.INJECTED):
    datasets = fetch_all_datasets(max_datasets=100, logger=logger)
    formatted = [format_dataset(ds) for ds in datasets]
    scitex.io.save(formatted, "openneuro_datasets.json")
    return 0
```

The SciTeX ecosystem follows the Four Freedoms for Research, inspired by [the Free Software Definition](https://www.gnu.org/philosophy/free-sw.en.html):

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere -- your machine, your terms.
>1. The freedom to **study** how every step works -- from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 -- because we believe research infrastructure deserves the same freedoms as the software it runs on.

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>

<!-- EOF -->
