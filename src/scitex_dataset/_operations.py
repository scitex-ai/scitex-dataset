#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_operations.py

"""The dataset operations as a Python API — one function per REST/MCP operation.

``scitex_dataset.datasets_list(...)`` is ``GET /api/v1/dataset/datasets`` is
the MCP tool ``dataset_datasets_list``: the REST handlers
(:mod:`scitex_dataset._rest._handlers`) are thin adapters over these
functions, so the three surfaces cannot drift apart.

A ``project`` here is a LOCAL project directory (the caller's own files); on
a host the REST layer resolves ``owner/slug`` to the authorized directory
first. Fetching into a hosted project is admitted by the host's storage gate
in the REST layer; locally there is no quota.

Example
-------
>>> import scitex as stx                                          # doctest: +SKIP
>>> hits = stx.dataset.datasets_list("eeg motor imagery", limit=5) # doctest: +SKIP
>>> stx.dataset.project_datasets_create(".", "reference:mnist-idx") # doctest: +SKIP
>>> stx.dataset.project_datasets_fetch(".", "reference:mnist-idx", wait=True) # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "cache_entries_get",
    "curated_datasets_get",
    "curated_datasets_list",
    "dataset_id",
    "datasets_get",
    "datasets_list",
    "jobs_get",
    "project_datasets_create",
    "project_datasets_fetch",
    "project_datasets_list",
]


class DatasetNotFound(LookupError):
    """No public dataset has this id."""


def dataset_id(key: str) -> str:
    """The one-segment id of a dataset key: ``/`` -> ``~``."""
    return key.replace("/", "~")


def _key(identifier: str) -> str:
    return (identifier or "").replace("~", "/")


def _cache(cache=None):
    from .cache import DatasetCache

    return cache if cache is not None else DatasetCache()


def _card(identifier: str, store=None) -> Dict[str, Any]:
    from . import catalog

    try:
        card = catalog.get_dataset(_key(identifier), store=store)
    except ValueError:
        card = None
    if card is None:
        raise DatasetNotFound(f"no such dataset: {identifier!r}")
    return card


def datasets_list(
    q: str = "",
    *,
    repository: Optional[str] = None,
    domain: Optional[str] = None,
    modality: Optional[str] = None,
    license: Optional[str] = None,
    min_size_bytes: Optional[int] = None,
    max_size_bytes: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    store=None,
) -> Dict[str, Any]:
    """Search openly available datasets across every supported repository.

    Returns ``{"results": [card...], "total", "index": {"status", ...}}``;
    see :func:`scitex_dataset.catalog.search_public`.
    """
    from . import catalog

    return catalog.search_public(
        q,
        repository=repository,
        domain=domain,
        modality=modality,
        license=license,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
        limit=limit,
        offset=offset,
        store=store,
    )


def datasets_get(dataset: str, *, store=None, cache=None) -> Dict[str, Any]:
    """One public dataset's card (metadata, files, licence, citation, snippet)."""
    card = _card(dataset, store)
    card["cache"] = (
        _cache(cache).status(card) if card.get("files") else {"state": "not-cached"}
    )
    return card


def cache_entries_get(dataset: str, *, store=None, cache=None) -> Dict[str, Any]:
    """The shared-cache state of a dataset: cached, partial or not cached."""
    card = _card(dataset, store)
    status = (
        _cache(cache).status(card) if card.get("files") else {"state": "not-cached"}
    )
    return {"dataset": card["key"], **status}


def project_datasets_list(
    project, *, store=None, record_store=None, cache=None
) -> List[Dict[str, Any]]:
    """The public datasets the project at ``project`` uses (see ``project_datasets``)."""
    from .project import project_datasets

    return project_datasets(
        project, store=store, record_store=record_store, cache=_cache(cache)
    )["datasets"]


def project_datasets_create(
    project,
    dataset: str,
    local_dir: Optional[str] = None,
    *,
    added_by: Optional[str] = None,
    store=None,
) -> Dict[str, Any]:
    """Use a public dataset in a project: pin version, DOI, licence, checksums."""
    from .project import record_dataset

    return record_dataset(
        project, _card(dataset, store), local_dir=local_dir, added_by=added_by
    )


def project_datasets_fetch(
    project,
    dataset: str,
    local_dir: Optional[str] = None,
    *,
    added_by: Optional[str] = None,
    wait: bool = False,
    store=None,
    cache=None,
) -> Dict[str, Any]:
    """Fetch a dataset into the project through the shared cache (a job)."""
    from .project import start_fetch

    return start_fetch(
        project,
        _card(dataset, store),
        local_dir=local_dir,
        added_by=added_by,
        cache=_cache(cache),
        wait=wait,
    )


def jobs_get(project, job: str) -> Dict[str, Any]:
    """Progress of a fetch job of the project."""
    from .project import job_status

    return job_status(project, job)


def curated_datasets_list(dataset_root=None) -> List[Dict[str, Any]]:
    """Curated AI-for-science datasets prepared under ``dataset_root``."""
    from . import curated

    return curated.list_curated(dataset_root)


def curated_datasets_get(
    benchmark: str, dataset_root=None, *, operator: bool = False
) -> Dict[str, Any]:
    """A curated dataset's source, derivation chain and masking summary.

    ``operator=True`` adds the NAMES of held-out files; content is never
    returned.
    """
    from . import curated

    try:
        described = curated.describe(benchmark, dataset_root, operator=operator)
    except KeyError:
        described = None
    if described is None:
        raise DatasetNotFound(f"no such curated dataset: {benchmark!r}")
    return described


# EOF
