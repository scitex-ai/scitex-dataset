#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/__init__.py

"""The PUBLIC datasets one research project uses: record, fetch, trace.

Everything here takes an explicit project ``root`` and never decides who may
call it: authorization — and whether a write is affordable — belongs to the
host (the Hub's project model and spend controls, reached through
``scitex_sdk.host`` by the Django app in :mod:`scitex_dataset._django`). That
split keeps the logic testable without a web stack and reusable from the CLI
or a script.

* :func:`record_dataset` pins a chosen public dataset in the project's
  manifest ``.scitex/dataset/sources.json`` (version, DOI, licence, citation,
  files with SHA-256).
* :func:`start_fetch` brings it into ``data/<name>/`` as a background job,
  through the shared cache (hard links, no second copy).
* :func:`project_datasets` lists what the project uses — recorded datasets
  and pinned public bytes detected on disk — with cache and verification
  state and, through the provider-neutral provenance contract
  (:mod:`.provenance`), the scripts that read them.

The project's OWN files are not datasets and are not listed here; they belong
to the host's file browser.
"""

from __future__ import annotations

from ._fetch import JOBS_RELPATH, job_status, start_fetch
from ._listing import project_datasets
from ._paths import DATA_DIR, DatasetError
from ._registry import REGISTRY_RELPATH, load_registry
from ._sources import (
    SOURCES_RELPATH,
    default_local_dir,
    detect_datasets,
    load_sources,
    pending_migrations,
    record_dataset,
)
from .provenance import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    ENTRY_POINT_GROUP,
    readers_of,
)

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "DATA_DIR",
    "ENTRY_POINT_GROUP",
    "JOBS_RELPATH",
    "REGISTRY_RELPATH",
    "SOURCES_RELPATH",
    "DatasetError",
    "default_local_dir",
    "detect_datasets",
    "job_status",
    "load_registry",
    "load_sources",
    "pending_migrations",
    "project_datasets",
    "readers_of",
    "record_dataset",
    "start_fetch",
]

# EOF
