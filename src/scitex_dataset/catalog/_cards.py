#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/catalog/_cards.py

"""One public dataset as a CARD: the uniform shape every interface shows.

The fetchers each return their repository's own dictionary (OpenNeuro's
``how_to_acknowledge``, Zenodo's ``size_bytes``, PhysioNet's ``size_gb``...).
A person choosing a dataset needs the same few facts for all of them —
title, repository, DOI or accession, licence, size, last update, how to cite —
so :func:`card_from_record` reads whichever of those a record carries and
never invents one: an absent licence is ``None``, shown as "not stated", not a
guess.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .._sources import SOURCE_INFO

__all__ = [
    "FETCHABLE_REPOSITORIES",
    "card_from_record",
    "citation_for",
    "human_size",
    "key_of",
    "reference_card",
    "snippet_for",
    "split_key",
]

#: Repositories :func:`scitex_dataset.download_dataset` can fetch file BYTES
#: from today. The others are catalogue-only: their metadata is searchable,
#: their files are not yet downloadable through the package.
FETCHABLE_REPOSITORIES = ("reference", "gin", "huggingface")

_UNITS = ("B", "kB", "MB", "GB", "TB", "PB")
_GIB = 1024**3
_DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)
_DESCRIPTION_CHARS = 600


def human_size(n_bytes: Optional[float]) -> Optional[str]:
    """``11490434`` -> ``"11.5 MB"`` (decimal units); ``None`` stays ``None``."""
    if n_bytes is None:
        return None
    try:
        value = float(n_bytes)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    for unit in _UNITS:
        if value < 1000 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000
    return None  # pragma: no cover - loop always returns


def key_of(repository: str, accession: str) -> str:
    """The card key ``<repository>:<accession>`` (the index's row id)."""
    return f"{repository}:{accession}"


def split_key(key: str) -> tuple[str, str]:
    """``"openneuro:ds004362"`` -> ``("openneuro", "ds004362")``."""
    repository, sep, accession = (key or "").partition(":")
    if not sep or not repository or not accession:
        raise ValueError(f"not a dataset key: {key!r} (expected <repository>:<id>)")
    return repository, accession


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("name") or value.get("id") or ""
    text = str(value).strip()
    return text or None


_RULE_RE = re.compile(r"[—–\-=_*#~]{3,}")
_TAG_RE = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    """Prose for a card: no markup, rule lines or runs of whitespace."""
    import html

    text = html.unescape(_TAG_RE.sub(" ", text))
    return " ".join(_RULE_RE.sub(" ", text).split())


def _authors(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    out = []
    for item in value if isinstance(value, Iterable) else []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("full_name") or ""
        else:
            name = str(item)
        if name.strip():
            out.append(name.strip())
    return out


def _doi(record: Dict[str, Any]) -> Optional[str]:
    raw = _text(record.get("doi"))
    if not raw:
        return None
    found = _DOI_RE.search(raw)
    return found.group(1).rstrip(".") if found else None


def _size_bytes(record: Dict[str, Any]) -> Optional[int]:
    size = record.get("size_bytes")
    if isinstance(size, (int, float)) and size > 0:
        return int(size)
    size_gb = record.get("size_gb")
    if isinstance(size_gb, (int, float)) and size_gb > 0:
        return int(size_gb * _GIB)
    return None


def _year(*values: Any) -> Optional[str]:
    for value in values:
        text = _text(value)
        if text and re.match(r"^\d{4}", text):
            return text[:4]
    return None


def _files(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The record's own file list, when the repository supplies one."""
    rows = []
    for item in record.get("files") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("key") or item.get("filename")
        if not path:
            continue
        size = item.get("size_bytes", item.get("size"))
        size = int(size) if isinstance(size, (int, float)) else None
        rows.append(
            {
                "path": str(path),
                "size_bytes": size,
                "size": human_size(size),
                "sha256": item.get("sha256"),
                "checksum": item.get("checksum"),
                "url": item.get("url"),
            }
        )
    return rows


def citation_for(card: Dict[str, Any]) -> str:
    """How to cite: the repository's own text when given, else built from facts."""
    if card.get("citation_text"):
        return card["citation_text"]
    authors = card.get("authors") or []
    who = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
    parts = [p for p in (who, f"({card['year']})" if card.get("year") else "") if p]
    head = " ".join(parts)
    title = card.get("title") or card.get("accession")
    version = f" (version {card['version']})" if card.get("version") else ""
    tail = (
        f"https://doi.org/{card['doi']}" if card.get("doi") else (card.get("url") or "")
    )
    pieces = [
        head,
        f"{title}{version}.",
        f"{card.get('repository_name')}.",
        tail,
    ]
    return " ".join(p for p in pieces if p).strip()


def _slug(accession: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", accession.split("/")[-1]).strip("-.")
    return slug or "dataset"


def snippet_for(card: Dict[str, Any]) -> str:
    """A ready-to-paste project-script snippet that fetches this dataset."""
    repository, accession = card["repository"], card["accession"]
    dest = card.get("local_dir") or f"./data/{_slug(accession)}"
    if repository == "reference":
        return (
            "import scitex as stx\n\n"
            f'rec = stx.dataset.fetch_reference("{accession}", "{dest}")\n'
            "print(rec.path, rec.sha256, rec.license)"
        )
    if repository in FETCHABLE_REPOSITORIES:
        return (
            "import scitex as stx\n\n"
            f'stx.dataset.download_dataset("{repository}", "{accession}", "{dest}")'
        )
    lookup = card.get("doi") or accession
    return (
        "import scitex as stx\n\n"
        f"# {card.get('repository_name')} is searchable but not yet downloadable\n"
        "# through scitex-dataset; this pins the record you chose.\n"
        f'rows = stx.dataset.db_search("{accession}", source="{repository}", limit=1)\n'
        f"# then download from {card.get('url') or 'the repository'} ({lookup})"
    )


def card_from_record(record: Dict[str, Any], repository: str) -> Dict[str, Any]:
    """Normalise one fetcher record of ``repository`` into a card."""
    info = SOURCE_INFO.get(repository, {})
    accession = str(record.get("id") or "").strip()
    size = _size_bytes(record)
    description = _plain(_text(record.get("description") or record.get("readme")) or "")
    if len(description) > _DESCRIPTION_CHARS:
        description = description[:_DESCRIPTION_CHARS].rstrip() + "…"
    doi = _doi(record)
    key = key_of(repository, accession)
    card = {
        # The REST id: the key as ONE path segment ("/" -> "~").
        "id": key.replace("/", "~"),
        "key": key,
        "repository": repository,
        "repository_name": info.get("name", repository),
        "domain": info.get("domain"),
        "accession": accession,
        "title": _text(record.get("name")) or accession,
        "description": description,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
        "url": _text(record.get("url")) or _text(record.get("homepage")),
        "license": _text(record.get("license")),
        "size_bytes": size,
        "size": human_size(size),
        "updated": _text(record.get("modified"))
        or _text(record.get("publish_date"))
        or _text(record.get("created")),
        "version": _text(record.get("version")),
        "authors": _authors(record.get("authors")),
        "year": _year(
            record.get("publish_date"), record.get("created"), record.get("modified")
        ),
        "modalities": [str(m) for m in (record.get("modalities") or []) if m],
        "n_subjects": record.get("n_subjects") or None,
        "files": _files(record),
        "citation_text": _text(record.get("citation"))
        or _text(record.get("how_to_acknowledge")),
        "fetchable": repository in FETCHABLE_REPOSITORIES,
    }
    card["citation"] = citation_for(card)
    card["snippet"] = snippet_for(card)
    return card


def _pinned_version(files: List[Dict[str, Any]]) -> str:
    """``sha256:<12 hex>`` naming the exact pinned bytes.

    One file: its own digest. Several: the digest of their sorted
    ``path:sha256`` lines, so the version changes iff any pinned byte does.
    """
    import hashlib

    if len(files) == 1:
        return f"sha256:{files[0]['sha256'][:12]}"
    lines = "\n".join(sorted(f"{f['path']}:{f['sha256']}" for f in files))
    return f"sha256:{hashlib.sha256(lines.encode()).hexdigest()[:12]}"


def reference_card(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A card for one entry of :data:`scitex_dataset.general.reference.REFERENCES`."""
    from ..general.reference import reference_files

    files = reference_files(entry)
    sizes = [f.get("size_bytes") for f in files]
    total = sum(sizes) if all(isinstance(s, int) for s in sizes) else None
    record = {
        "id": entry["id"],
        "name": entry.get("name"),
        "description": entry.get("description"),
        "license": entry.get("license"),
        "citation": entry.get("citation"),
        "homepage": entry.get("homepage"),
        # A reference file is versioned by its pinned bytes: the sha256 IS
        # the version unless the entry names an upstream release.
        "version": entry.get("version") or _pinned_version(files),
        "size_bytes": total,
        "files": files,
        # Only a DOI the entry states for the DATA; the citation's DOI names
        # the paper, and showing it as the dataset's would mislead.
        "doi": entry.get("doi"),
    }
    card = card_from_record(record, "reference")
    card["format"] = entry.get("format")
    return card


# EOF
