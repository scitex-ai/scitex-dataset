#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/database.py

"""The dataset index, held in the fleet's shared store.

Usage:
    >>> from scitex_dataset import database as db
    >>> db.build()  # Fetch all sources and index them
    >>> results = db.search("alzheimer EEG", min_subjects=20)

WHAT CHANGED, AND WHY IT IS NOT A PORT
--------------------------------------
This module used to keep its own private database file with its own
full-text index, its own schema and its own triggers. That was a second
storage engine living inside a leaf package, which is the shape the fleet
ruled out: state has exactly one home, and a private file is not it. A file
also has no notion of WHO — anyone who can open it holds every permission —
so an index built that way can be handed over but never collaborated on.

The index now lives in :mod:`scitex_dev.store`, used DIRECTLY. There is no
``scitex_dataset`` database layer between this module and the primitive and
there is not meant to be one: a wrapper is how two packages end up with two
answers to the same question. What stays here is the DOMAIN — the shape
lives in :mod:`._index_schema`, the verbs live below.

THE ONE THING THE PRIMITIVE DID NOT HAVE was a way to read by criteria
rather than by key, which is a large part of why a private index looked
reasonable. Rather than rebuild one here, that surface was added to the
store (``Store.search`` / ``count`` / ``tally``, ``Query``), where every
package gets it. If something here still feels awkward, that is a gap in
the primitive and belongs there too.

NOTHING IS DELETED ANY MORE
---------------------------
The store has no delete verb; :func:`clear` hides rows instead. A hidden
row leaves the default view — so a search after ``clear`` finds nothing,
exactly as before — but the record and its history stay readable, and a
rebuild revives it rather than starting it over.
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

from scitex_dev.decorators import supports_return_as
from scitex_dev.store import (
    ANY_REVISION,
    Query,
    Store,
    WriterPolicy,
    contains,
    either,
    eq,
    gte,
    host_store,
    lte,
    nonempty,
)

from ._index_schema import PKG, SCHEMA, VALID_ORDERS, row_values

__all__ = [
    "build",
    "clear",
    "get_stats",
    "get_store",
    "search",
    "store_description",
    "update",
]

#: Which module holds each catalogue source's fetchers. A table rather than
#: a chain of ``elif``, so importing one source's HTTP client does not drag
#: in the other ten.
_SOURCE_MODULES = {
    "openneuro": "neuroscience.openneuro",
    "dandi": "neuroscience.dandi",
    "physionet": "neuroscience.physionet",
    "zenodo": "general.zenodo",
    "figshare": "general.figshare",
    "openml": "general.openml",
    "geo": "biology.geo",
    "chembl": "pharmacology.chembl",
    "moleculenet": "pharmacology.moleculenet",
    "clinicaltrials": "medical.clinicaltrials",
    "huggingface": "general.huggingface",
}

_STORE: "Store | None" = None


def get_store(store: "Store | None" = None) -> Store:
    """The index store — this host's, unless one is handed in.

    ``store`` is how a test points at a throwaway schema without patching
    anything: pass a :class:`~scitex_dev.store.Store` and it is used
    verbatim. Every public function here takes the same argument for the
    same reason.

    ``host_store`` is the single resolver for WHERE that is; nothing in
    this package reads a connection string. The result is cached because
    opening a store creates its tables under an advisory lock, and a CLI
    doing that per subcommand would pay for it every time.
    """
    global _STORE
    if store is not None:
        return store
    if _STORE is None:
        _STORE = Store(
            host_store(pkg=PKG, name="index"),
            SCHEMA,
            node=socket.gethostname(),
            writer_policy=WriterPolicy.MULTI_WRITER,
        )
    return _STORE


def store_description(store: "Store | None" = None) -> str:
    """A credential-free one-line description of where the index lives.

    Replaces the old path accessor. There is no file any more, and
    returning a path would be a lie a caller could act on — the CLI printed
    it beside a file size, and both facts stopped existing at once.

    RESOLVES WITHOUT CONNECTING when no store is open yet. Asking where the
    index WOULD be must not create it: ``db build --dry-run`` prints this
    line, and going through :func:`get_store` would open a connection and
    run the schema DDL — a dry run whose whole promise is that it writes
    nothing.
    """
    if store is not None:
        return store.target.describe()
    if _STORE is not None:
        return _STORE.target.describe()
    return host_store(pkg=PKG, name="index").describe()


def _fetchers(source: str):
    """``(fetch_all_datasets, format_dataset)`` for one source, or None."""
    if source not in _SOURCE_MODULES:
        return None
    from importlib import import_module

    module = import_module(f".{_SOURCE_MODULES[source]}", package=__package__)
    return module.fetch_all_datasets, module.format_dataset


@supports_return_as
def build(
    sources: Optional[List[str]] = None,
    store: "Store | None" = None,
    logger=None,
) -> Dict[str, int]:
    """Build the index from all sources.

    Parameters
    ----------
    sources : list, optional
        Sources to fetch: ["openneuro", "dandi", "physionet"].
        Default: all catalog sources.
    store : Store, optional
        The store to write into. Default: this host's.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    dict
        Count of datasets indexed per source.
    """
    if sources is None:
        from ._sources import CATALOG_SOURCES

        # HuggingFace is NOT included by default — its catalog is unbounded
        # and would dominate the index. Pass `sources=["huggingface", ...]`
        # explicitly to opt in (uses query="" + max=1000 cap).
        sources = list(CATALOG_SOURCES)

    target = get_store(store)
    counts: Dict[str, int] = {}

    for source in sources:
        if logger:
            logger.info(f"Fetching from {source}...")

        fetchers = _fetchers(source)
        if fetchers is None:
            if logger:
                logger.warning(f"Unknown source: {source}")
            continue

        fetch_all_datasets, format_dataset = fetchers
        try:
            raw = fetch_all_datasets(logger=logger)
            datasets = [format_dataset(ds) for ds in raw]

            # One transaction per source rather than one per row. A logical
            # write is three statements and therefore three durable commits,
            # which is what dominates a bulk load.
            with target.batch():
                for dataset in datasets:
                    # ANY_REVISION rather than read-then-compare: a rebuild
                    # re-states the whole record from upstream, so there is
                    # no local edit for a concurrent writer to lose. It also
                    # revives a row a previous `clear` retired, which is
                    # what "rebuild" ought to mean.
                    target.put(
                        row_values(dataset, source),
                        expected_revision=ANY_REVISION,
                    )

            counts[source] = len(datasets)

            if logger:
                logger.info(f"Indexed {len(datasets)} from {source}")

        except Exception as exc:
            if logger:
                logger.error(f"Error fetching {source}: {exc}")
            counts[source] = 0

    return counts


def update(
    source: str,
    store: "Store | None" = None,
    logger=None,
) -> int:
    """Update a single source in the index.

    Parameters
    ----------
    source : str
        Source to update: "openneuro", "dandi", or "physionet".
    store : Store, optional
        The store to write into. Default: this host's.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    int
        Number of datasets indexed.
    """
    result = build(sources=[source], store=store, logger=logger)
    return result.get(source, 0)


def _as_query(
    query: Optional[str],
    source: Optional[str],
    modality: Optional[str],
    min_subjects: Optional[int],
    max_subjects: Optional[int],
    min_downloads: Optional[int],
    has_readme: bool,
    limit: int,
    offset: int,
    order_by: str,
) -> Query:
    """Translate the caller's filters into one store query."""
    criteria = []
    if source:
        criteria.append(eq("source", source))
    if modality:
        # Either the modality LIST holds it, or it IS the primary one.
        # `contains` asks the JSON document, so a dataset whose prose
        # mentions "eeg" is not swept in the way a substring match on the
        # serialised column would sweep it.
        criteria.append(
            either(contains("modalities", modality), eq("primary_modality", modality))
        )
    if min_subjects is not None:
        criteria.append(gte("n_subjects", min_subjects))
    if max_subjects is not None:
        criteria.append(lte("n_subjects", max_subjects))
    if min_downloads is not None:
        criteria.append(gte("downloads", min_downloads))
    if has_readme:
        criteria.append(nonempty("readme"))

    ordering = order_by if order_by in VALID_ORDERS else "downloads"
    return (
        Query()
        .matching(query)
        .where(*criteria)
        .ordered_by(ordering)
        .limited(limit, offset=offset)
    )


@supports_return_as
def search(
    query: Optional[str] = None,
    source: Optional[str] = None,
    modality: Optional[str] = None,
    min_subjects: Optional[int] = None,
    max_subjects: Optional[int] = None,
    min_downloads: Optional[int] = None,
    has_readme: bool = False,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "downloads",
    store: "Store | None" = None,
) -> List[Dict[str, Any]]:
    """Search the index.

    Parameters
    ----------
    query : str, optional
        Full-text search over id, name, readme and tasks. Bare words are
        ANDed, ``"quoted phrases"`` are phrases, ``or`` disjoins and a
        leading ``-`` negates. Malformed input matches nothing rather than
        raising, because the text came from a person.

        WORDS ARE STEMMED and English stopwords are dropped, which the
        previous tokeniser did neither of. So ``studies`` now finds a
        dataset that says ``study`` — a superset, and the reason to prefer
        it — but a query consisting only of stopwords (``the``, ``of``)
        matches nothing rather than everything. Pass ``text_config="simple"``
        on the schema if a package ever needs the literal behaviour back.
    source : str, optional
        Filter by source: "openneuro", "dandi", "physionet".
    modality : str, optional
        Filter by modality (e.g., "mri", "eeg").
    min_subjects : int, optional
        Minimum number of subjects.
    max_subjects : int, optional
        Maximum number of subjects.
    min_downloads : int, optional
        Minimum download count.
    has_readme : bool
        Only include datasets with a non-empty readme.
    limit : int
        Maximum results (default: 50).
    offset : int
        Skip first N results (for pagination).
    order_by : str
        Order by: downloads, views, n_subjects, size_gb, name, created.
    store : Store, optional
        The store to read from. Default: this host's.

    Returns
    -------
    list
        Matching datasets, in the shape the fetchers produced.
    """
    target = get_store(store)
    found = target.search(
        _as_query(
            query,
            source,
            modality,
            min_subjects,
            max_subjects,
            min_downloads,
            has_readme,
            limit,
            offset,
            order_by,
        )
    )
    return [row.values["record"] for row in found]


@supports_return_as
def get_stats(store: "Store | None" = None) -> Dict[str, Any]:
    """Index statistics.

    Returns
    -------
    dict
        Counts per source, the total, where the store is, and when the
        index was last written.

    ``exists`` reports whether anything is INDEXED, not whether a file is on
    disk. The store always exists; an empty one is the state that used to be
    "no database yet", and it is the one a caller has to act on.
    """
    target = get_store(store)
    total = target.count()

    if not total:
        return {
            "exists": False,
            "message": "Index not built. Run: db.build()",
            "store": target.target.describe(),
        }

    newest = target.search(Query().ordered_by("indexed_at").limited(1))
    return {
        "exists": True,
        "store": target.target.describe(),
        "total_datasets": total,
        "by_source": target.tally("source"),
        "last_build": newest[0].values["indexed_at"] if newest else None,
    }


def clear(store: "Store | None" = None) -> bool:
    """Retire every indexed dataset.

    Returns
    -------
    bool
        True if anything was retired, False if the index was already empty.

    NOT A DELETE, and the difference is the point. The store has no delete
    verb: the failure it exists to make impossible is a row disappearing
    because some code decided it should. So this HIDES. The datasets leave
    the default view, so :func:`search` and :func:`get_stats` answer exactly
    as they did when the old file had been removed, while the records and
    their history stay readable and a later :func:`build` brings them back.
    """
    target = get_store(store)
    visible = target.search(Query())
    for row in visible:
        target.hide({"id": row.values["id"]}, expected_revision=ANY_REVISION)
    return bool(visible)

# EOF
