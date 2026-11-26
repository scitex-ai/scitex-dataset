#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/__init__.py

"""The datasets of ONE research project: inventory, registration, provenance.

Everything here takes an explicit project ``root`` and never decides who may
call it: authorization belongs to the host (the Hub's project model, reached
through ``scitex_sdk.host.project_access`` by the Django app in
:mod:`scitex_dataset._django`). That split keeps the logic testable without a
web stack and reusable from the CLI or a script.

A project's datasets are

* the files under its ``data/`` directory (the scitex-template layout), and
* datasets it REGISTERED explicitly — a path already inside the project, or an
  external URL — kept in ``<root>/.scitex/dataset/registry.json``.

Each entry carries its size and SHA-256 when it is a local file, and, when a
provenance provider is installed, the scripts that READ it (see
:mod:`scitex_dataset.project.provenance`, a versioned, provider-neutral
contract — no provider is imported here).
"""

from __future__ import annotations

from ._inventory import (
    DATA_DIR,
    MAX_HASH_BYTES,
    MAX_LISTED_FILES,
    DatasetError,
    list_datasets,
    register_dataset,
    save_upload,
)
from ._registry import REGISTRY_RELPATH, load_registry
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
    "MAX_HASH_BYTES",
    "MAX_LISTED_FILES",
    "REGISTRY_RELPATH",
    "DatasetError",
    "list_datasets",
    "load_registry",
    "readers_of",
    "register_dataset",
    "save_upload",
]

# EOF
