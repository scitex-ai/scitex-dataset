#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/catalog/__init__.py

"""Find OPENLY AVAILABLE datasets across every supported repository.

The browser-facing read side of the package, built only from what already
exists: the metadata index in the shared SciTeX store
(:mod:`scitex_dataset.database`, filled by each repository's fetcher) and the
pinned reference-file registry (:mod:`scitex_dataset.general.reference`).
Nothing here talks to a repository; the fetchers stay the only code that does.

Every result is a CARD (:mod:`._cards`): title, repository, DOI or accession,
licence, size (human-readable), last update and a citation, in one shape for
all repositories.

The index can be unavailable on a host (never built, or the store refuses the
connection). That is reported in the result's ``index`` block — never hidden
behind an empty list, which would read as "no dataset matches".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .._sources import ALL_SOURCES, DOMAINS, SOURCE_INFO, sources_in_domain
from ._cards import (
    FETCHABLE_REPOSITORIES,
    card_from_record,
    citation_for,
    human_size,
    key_of,
    reference_card,
    snippet_for,
    split_key,
)

__all__ = [
    "FETCHABLE_REPOSITORIES",
    "SEARCHABLE_REPOSITORIES",
    "card_from_record",
    "citation_for",
    "get_dataset",
    "human_size",
    "key_of",
    "reference_card",
    "repositories",
    "search_public",
    "snippet_for",
    "split_key",
]

#: Repositories a public-dataset search covers: the catalogue sources, the
#: on-demand HuggingFace Hub, and the reference files. The AI-for-science
#: benchmarks are CURATED derivatives, shown by :mod:`scitex_dataset.curated`.
SEARCHABLE_REPOSITORIES = tuple(
    s for s in ALL_SOURCES if SOURCE_INFO[s]["kind"] != "agentic-bench"
)

_GIB = 1024**3


def _resolve(store):
    """A Store, ``None`` (the host's), or a zero-argument factory of one.

    A factory is opened only when the index is actually needed, so a lookup
    of a reference file never touches the store.
    """
    if store is not None and callable(store) and not hasattr(store, "search"):
        return store()
    return store


def repositories() -> List[Dict[str, Any]]:
    """Every searchable repository with its display name, domain and fetchability."""
    return [
        {
            "id": source,
            "name": SOURCE_INFO[source]["name"],
            "domain": SOURCE_INFO[source]["domain"],
            "description": SOURCE_INFO[source]["description"],
            "homepage": SOURCE_INFO[source]["url"],
            "fetchable": source in FETCHABLE_REPOSITORIES,
        }
        for source in SEARCHABLE_REPOSITORIES
    ]


def _reference_cards() -> List[Dict[str, Any]]:
    from ..general.reference import list_references

    return [reference_card(entry) for entry in list_references()]


def _words(text: Optional[str]) -> List[str]:
    return [w for w in (text or "").lower().split() if w]


def _reference_matches(
    card: Dict[str, Any], words: List[str], license: Optional[str]
) -> bool:
    if license and card.get("license") != license:
        return False
    haystack = " ".join(
        str(card.get(k) or "")
        for k in ("accession", "title", "description", "citation_text", "format")
    ).lower()
    return all(w in haystack for w in words)


def _within(card: Dict[str, Any], min_bytes, max_bytes) -> bool:
    size = card.get("size_bytes")
    if min_bytes is not None and (size is None or size < min_bytes):
        return False
    if max_bytes is not None and (size is None or size > max_bytes):
        return False
    return True


def _wanted_sources(
    repository: Optional[str], domain: Optional[str]
) -> Optional[List[str]]:
    wanted = set(SEARCHABLE_REPOSITORIES)
    if repository:
        if repository not in wanted:
            raise ValueError(f"unknown repository {repository!r}")
        wanted &= {repository}
    if domain:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}")
        wanted &= set(sources_in_domain(domain))
    if not repository and not domain:
        return None
    return sorted(wanted)


def search_public(
    query: str = "",
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
    """Search public datasets; returns cards plus the index's state and facets.

    Returns::

        {"query", "results": [card...], "total": int,
         "index": {"status": "ok" | "unavailable", "error": str | None,
                   "facets": {"source": {...}, "license": {...}}}}

    Reference files are matched in memory (they are a registry, not an index
    row) and listed first; the index supplies the rest, paged by ``limit`` /
    ``offset``.
    """
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    sources = _wanted_sources(repository, domain)
    words = _words(query)

    references: List[Dict[str, Any]] = []
    if (sources is None or "reference" in sources) and not modality:
        references = [
            c
            for c in _reference_cards()
            if _reference_matches(c, words, license)
            and _within(c, min_size_bytes, max_size_bytes)
        ]

    index: Dict[str, Any] = {"status": "ok", "error": None, "facets": {}}
    results: List[Dict[str, Any]] = list(references) if offset == 0 else []
    total = len(references)
    index_sources = (
        None if sources is None else [s for s in sources if s != "reference"]
    )
    if index_sources is None or index_sources:
        from .. import database

        try:
            store = _resolve(store)
            page = database.search_page(
                query or None,
                sources=index_sources,
                modality=modality or None,
                license=license or None,
                min_size_gb=None if min_size_bytes is None else min_size_bytes / _GIB,
                max_size_gb=None if max_size_bytes is None else max_size_bytes / _GIB,
                limit=limit,
                offset=offset,
                store=store,
            )
            index["facets"] = database.facets(store=store)
        except Exception as exc:  # noqa: BLE001 - the store is a remote boundary
            index = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "facets": {},
            }
        else:
            total += page["total"]
            results.extend(card_from_record(rec, src) for src, rec in page["items"])
    return {
        "query": query,
        "results": results,
        "total": total,
        "limit": limit,
        "offset": offset,
        "index": index,
    }


def get_dataset(key: str, store=None) -> Optional[Dict[str, Any]]:
    """The card of ``key`` (``<repository>:<accession>``), or ``None``.

    Raises :class:`ValueError` for a malformed key or an unknown repository,
    and lets a store failure propagate: "the index is down" is not "absent".
    """
    repository, accession = split_key(key)
    if repository not in SEARCHABLE_REPOSITORIES:
        raise ValueError(f"unknown repository {repository!r}")
    if repository == "reference":
        for card in _reference_cards():
            if card["accession"] == accession.lower():
                return card
        return None
    from .. import database

    record = database.get_record(repository, accession, store=_resolve(store))
    return None if record is None else card_from_record(record, repository)


# EOF
