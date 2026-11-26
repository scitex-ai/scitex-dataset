---
description: "[TOPIC] A project's datasets and the Hub Dataset app. [DETAILS] List data/ files and registered datasets with size and SHA-256, upload or register a path/URL, and read which scripts read each dataset through the provider-neutral provenance-readers contract."
tags: [scitex-dataset-project-datasets]
---
# Project datasets and the Dataset app

`scitex_dataset.project` works on ONE project root; authorization is the
caller's (the Hub app asks the host through `scitex_sdk.host.project_access`).

```python
from scitex_dataset.project import list_datasets, register_dataset, save_upload

listing = list_datasets("/path/to/project")   # files under data/ + registered
register_dataset("/path/to/project", "data/raw/session-01", name="Session 1")
register_dataset("/path/to/project", "https://doi.org/10.5281/zenodo.1", kind="url")
save_upload("/path/to/project", "b.csv", [b"1,2\n"], subdir="raw")  # never overwrites
```

- Registrations live in `<project>/.scitex/dataset/registry.json`; a path
  entry records size and SHA-256 at registration, and the listing reports
  `unchanged`, `changed`, `missing` or `unverified`. URLs are never fetched.
- Paths are confined to the project (symlinks resolved); files above
  `MAX_HASH_BYTES` are listed with a reason instead of a checksum.

## Hub app

`pip install 'scitex-dataset[server]'` registers the `scitex.apps` entry point
`dataset`; a SciTeX host mounts it at `/apps/dataset/?project=owner/slug`.
Members read, members with write permission upload/register, others get 404.
JSON API: `GET api/datasets`, `POST api/upload` (`file`, `subdir`),
`POST api/register` (`path` or `url`, `name`, `note`).

## Provenance hook (contract `scitex-dataset.provenance-readers` 1.0.0)

A provider registers in the `scitex.dataset.provenance` entry-point group an
object with `name`, `contract_version` and
`readers(project_root, paths) -> {path: [{"script", "run_id", "started_at", "status"}]}`.
The schema ships as `scitex_dataset/project/provenance-readers.schema.1.0.0.json`.
scitex-clew provides one (`scitex_clew._dataset_readers:ReadersEntry`); the
dataset package never imports a provider.
