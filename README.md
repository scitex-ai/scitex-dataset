# SciTeX Dataset (<code>scitex-dataset</code>)

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Unified access to neuroscience and scientific datasets</b></p>

<p align="center">
  <a href="https://scitex-dataset.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-dataset</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-dataset/"><img src="https://img.shields.io/pypi/v/scitex-dataset.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-dataset/"><img src="https://img.shields.io/pypi/pyversions/scitex-dataset.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/install-test.yml"><img src="https://github.com/ywatanabe1989/scitex-dataset/actions/workflows/install-test.yml/badge.svg" alt="Install Test"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-dataset"><img src="https://codecov.io/gh/ywatanabe1989/scitex-dataset/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scitex-dataset.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/scitex-dataset/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Problem and Solution

| # | Problem | Solution |
|---|---------|----------|
| 1 | **Public dataset repositories balkanized** -- OpenNeuro (BIDS) + DANDI (NWB) + PhysioNet (WFDB) + Zenodo (generic) + GEO / ChEMBL / ClinicalTrials — different APIs, auth, download tools | **Unified fetcher** -- `stx.dataset.neuroscience.openneuro.fetch_all_datasets()` same call shape across all; local FTS5 search across metadata |
| 2 | **"Download this BIDS dataset" means reading DataLad docs first** -- the barrier is tooling, not knowledge | **One-line fetch** -- no DataLad setup; the module handles auth, resumption, checksums transparently |

## Supported repositories

| Domain | Repository | Description | Data Types |
|--------|------------|-------------|------------|
| neuroscience | **OpenNeuro** | Open BIDS neuroimaging platform | MRI, EEG, MEG, iEEG, PET |
| neuroscience | **DANDI** | BRAIN Initiative archive (NWB) | Electrophysiology, Ophys |
| neuroscience | **PhysioNet** | Physiological signal databases | ECG, EEG, clinical data |
| general | **Zenodo** | General scientific data (CERN) | Any research data |
| general | **Figshare** | Research data sharing platform | Any research data |
| general | **OpenML** | Machine-learning datasets | Tabular ML benchmarks |
| general | **HuggingFace Hub** | ML datasets / models (on-demand) | Any |
| biology | **GEO** | Gene Expression Omnibus (NCBI) | Transcriptomics, microarray |
| pharmacology | **MoleculeNet** | Molecular ML benchmark suite | SMILES, properties |
| pharmacology | **ChEMBL** | Bioactivity database (EBI) | IC50/Ki/EC50 assays |
| medical | **ClinicalTrials.gov** | NIH study registry | Trial metadata |

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

<details open>
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

> **[Full API reference](https://scitex-dataset.readthedocs.io/en/latest/api/scitex_dataset.html)**

</details>

<details>
<summary><strong>CLI Commands</strong></summary>

<br>

```bash
scitex-dataset --help-recursive             # Show all commands

# Grammar: scitex-dataset <domain> <dataset> <action>
scitex-dataset neuroscience openneuro fetch -n 100 -o datasets.json -v
scitex-dataset neuroscience dandi fetch -n 50 -o dandi.json -v
scitex-dataset neuroscience physionet fetch -n 50 -v
scitex-dataset general zenodo fetch -q "neuroscience" -n 20

# HuggingFace (general/huggingface noun-group has 4 verbs)
scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
scitex-dataset general huggingface search "biology" -n 20 --json

# Local database
scitex-dataset db build                     # index all catalog sources
scitex-dataset db search "epilepsy EEG"     # full-text search
scitex-dataset db show-stats                # show statistics

# Introspection
scitex-dataset list-python-apis -v          # list Python API tree
scitex-dataset mcp list-tools -v            # list MCP tools
```

> **[Full CLI reference](https://scitex-dataset.readthedocs.io/en/latest/quickstart.html)**

</details>

<details>
<summary><strong>MCP Server</strong></summary>

<br>

AI agents can discover and query neuroscience datasets autonomously.

| Tool | Description |
|------|-------------|
| `dataset_list_sources` | Enumerate the 11 supported sources |
| `dataset_filter_results` | Filter / rank fetched datasets in memory |
| `dataset_<src>_fetch` | One per catalog source (10 total) |
| `dataset_hf_fetch` / `_search` / `_info` / `_download_file` | HuggingFace family |
| `dataset_db_build` / `_search` / `_stats` | Local SQLite + FTS5 index |
| `dataset_skills_list` / `_get` | Bundled skill pages |

<sub><b>Table 2.</b> 21 MCP tools across catalog fetchers, HuggingFace,
the local index, and skill introspection. Every MCP tool has a matching
public Python alias (e.g. ``scitex_dataset.openneuro_fetch``).</sub>

```bash
scitex-dataset mcp start
```

> **[Full MCP specification](https://scitex-dataset.readthedocs.io/en/latest/api/scitex_dataset._mcp.html)**

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

`scitex-dataset` is part of [**SciTeX**](https://scitex.ai). Install via
the umbrella with `pip install scitex[dataset]` to use as
`scitex.dataset` (Python) or `scitex dataset ...` (CLI).

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
