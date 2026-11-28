---
description: "[TOPIC] Find open datasets and use them in a project; the Hub Dataset app and the dataset REST/MCP API. [DETAILS] Search public datasets across every repository, pin a dataset (version, DOI, licence, checksums) in .scitex/dataset/sources.json, fetch through the shared cache, list a project's public datasets with the scripts that read them, curated AI-for-science datasets, and the /api/v1/dataset operations."
tags: [scitex-dataset-project-datasets]
---
# Open datasets in a project, the Dataset app and the dataset API

One operation set, three surfaces — Python, REST (`/api/v1/dataset/`) and MCP
(`dataset_<resource>_<verb>`):

| Python (`stx.dataset.`) | REST | MCP tool |
|---|---|---|
| `datasets_list(q, repository=, domain=, modality=, license=, min_size_bytes=, max_size_bytes=)` | `GET datasets` | `dataset_datasets_list` |
| `datasets_get(dataset)` | `GET datasets/{dataset_id}` | `dataset_datasets_get` |
| `cache_entries_get(dataset)` | `GET cache-entries/{id}` | `dataset_cache_entries_get` |
| `project_datasets_list(project)` | `GET project-datasets?project=owner/slug` | `dataset_project_datasets_list` |
| `project_datasets_create(project, dataset, local_dir=)` | `POST project-datasets` | `dataset_project_datasets_create` |
| `project_datasets_fetch(project, dataset, wait=)` | `POST project-datasets/{id}/fetch` (202 job) | `dataset_project_datasets_fetch` |
| `jobs_get(project, job)` | `GET jobs/{job_id}?project=` | `dataset_jobs_get` |
| `curated_datasets_list()` / `curated_datasets_get(benchmark)` | `GET curated-datasets[/{id}]` | `dataset_curated_datasets_*` |

```python
import scitex as stx

hits = stx.dataset.datasets_list("eeg motor imagery", limit=5)["results"]
stx.dataset.project_datasets_create(".", "reference:mnist-idx")          # pins it
stx.dataset.project_datasets_fetch(".", "reference:mnist-idx", wait=True)  # cache -> data/mnist-idx
```

- A dataset id is `<repository>:<accession>` (`openneuro:ds004362`,
  `reference:mnist-idx`); `/` in an accession is written `~` in a URL.
- The project manifest `.scitex/dataset/sources.json` records available version, DOI,
  licence, citation and file checksums; the 0.15 `registry.json` URL
  registrations are migrated into it on the next write.
- Pinned public bytes already under `data/` are DETECTED (sha256 against the
  reference registry), so a project that downloaded MNIST by hand shows it
  as the MNIST dataset. Project outputs (`data/results`, figures) are never
  listed — they belong to the host's file browser.
- Fetching goes through the shared cache (`$SCITEX_DATASET_CACHE`, default
  `~/.scitex/dataset/cache`): downloaded once, hard-linked into each project;
  LRU eviction to `cache.max_bytes` of the dataset config.
- On a host, a fetch is admitted by the host's storage gate
  (`scitex_sdk.host.require_storage_write`: quota and spend controls); with no
  gate the host refuses to write (503), it never assumes.
- REST/MCP are declared once in `scitex_dataset._rest` (scitex-sdk
  `ApiOperation`s, entry point `scitex.apis: dataset`); a host mounts them with
  `scitex_sdk.app.api_serve.api_urlpatterns()` and no dataset code. Scopes are
  derived: `dataset:<resource>:<read|write>`.

Search coverage exceeds download coverage. A catalog record can lack a version,
licence or upstream digest. For pinned files the downloader checks the expected
SHA-256. For repository downloads that supply no digest it computes SHA-256 after
retrieval. To claim replay of a specific upstream revision, verify that the
adapter selected that revision and that the recorded bytes match.

## Which scripts read a dataset — provenance-readers 1.1.0

Providers (e.g. scitex-clew) register in the `scitex.dataset.provenance`
entry-point group. 1.1.0 adds `store=` — the host's routing to the project's
own record store (`scitex_sdk.host.StoreAccess` fields) — so a Hub project's
reads are found in its tenant store; and the result's `sources` names the
public dataset (key, version, DOI, sha256) of each path, so a claim traced
through a script ends at an exact public dataset version.

## Curated AI-for-science datasets

`curated_datasets_get(benchmark)` shows the source snapshot and version, the
derivation chain (raw -> standardize -> for_solver / eval) from
`MANIFEST.yaml`, the masking policies, and COUNTS of held-out content.
Held-out file names are for operators only (`operator=True`); content is
never returned by any surface.
