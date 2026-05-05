---
description: |
  [TOPIC] Data Sources
  [DETAILS] All 11 supported repositories + the local SQLite index.
tags: [scitex-dataset-data-sources, scitex-dataset]
---


# Data Sources

scitex-dataset supports **10 catalog sources** (enumerable, indexable
into the local SQLite + FTS5 cache) plus **HuggingFace Hub** (on-demand
fetch by repo_id; not indexed by default).

Every catalog source exposes the same two callables —
``fetch_all_datasets()`` and ``format_dataset(ds)`` — so they plug
uniformly into ``database.build()`` and ``search.search_datasets()``.

## Catalog sources

### OpenNeuro — BIDS neuroimaging
```python
from scitex_dataset.neuroscience.openneuro import fetch_all_datasets, format_dataset
datasets = [format_dataset(d) for d in fetch_all_datasets(max_datasets=50)]
```

### DANDI Archive — NWB neurophysiology
```python
from scitex_dataset.neuroscience import dandi
datasets = [dandi.format_dataset(d) for d in dandi.fetch_all_datasets(max_datasets=50)]
```

### PhysioNet — EEG / ECG / clinical waveforms
```python
from scitex_dataset.neuroscience import physionet
datasets = [physionet.format_dataset(d) for d in physionet.fetch_all_datasets()]
```

### Zenodo — general scientific data (CERN)
```python
from scitex_dataset.general import zenodo
datasets = zenodo.fetch_all_datasets(query="neural network")
```

### Figshare — general research data
```python
from scitex_dataset.general import figshare
datasets = figshare.fetch_all_datasets(query="biology")
```

### OpenML — ML datasets / benchmarks
```python
from scitex_dataset.general import openml
datasets = openml.fetch_all_datasets(max_datasets=100)
```

### MoleculeNet — molecular ML benchmarks
```python
from scitex_dataset.pharmacology import moleculenet
datasets = moleculenet.fetch_all_datasets()
```

### GEO — Gene Expression Omnibus (NCBI)
```python
from scitex_dataset.biology import geo
datasets = geo.fetch_all_datasets(max_datasets=100)
```

### ChEMBL — bioactivity (EBI)
```python
from scitex_dataset.pharmacology import chembl
datasets = chembl.fetch_all_datasets()
```

### ClinicalTrials.gov — interventional / observational studies
```python
from scitex_dataset.medical import clinicaltrials
datasets = clinicaltrials.fetch_all_datasets()
```

## On-demand source

### HuggingFace Hub — datasets and models

HF has no bounded catalog (millions of datasets), so it isn't indexed
by ``database.build()`` by default. Three workflows:

```python
from scitex_dataset.general import huggingface as hf

# 1) Search the live catalog by keyword
results = hf.search_hub("biology", limit=50)

# 2) Inspect a specific repo before downloading
info = hf.dataset_info("Anthropic/BioMysteryBench-full")

# 3) Snapshot-download (gated repos pick up tokens via HF_TOKEN /
#    HF_TOKEN_PATH / ~/.bash.d/secrets/access_tokens/huggingface.txt)
path = hf.fetch_dataset(
    "Anthropic/BioMysteryBench-full",
    local_dir="/data/gpfs/projects/punim2354/biomysterybench",
)
```

To opt HF into the local index explicitly (capped at 1000 items via
``search_hub``):

```python
from scitex_dataset import database
database.build(sources=["huggingface"])
```

CLI mirror: ``scitex-dataset general huggingface (fetch | search | info | download-file)``.

## Local Database

```python
from scitex_dataset import database as db
db.build()                    # Build/refresh (catalog sources only)
results = db.search("EEG")    # Search local cache
stats = db.get_stats()        # Database statistics
```
