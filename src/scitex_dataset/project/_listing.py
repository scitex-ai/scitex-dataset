#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_listing.py

"""The public datasets ONE project uses, ready to show.

Merges three sources of truth, each kept where it belongs:

* the project's manifest (``sources.json``): what the project RECORDED;
* :func:`._sources.detect_datasets`: pinned public bytes already on disk,
  recorded or not;
* the shared cache, the project's files and the provenance providers: the
  cache state, whether the local copy still verifies, and which scripts read
  the dataset's files.

Project outputs (``data/results``, figures, literature) are never listed:
only files that belong to a public dataset are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..catalog import get_dataset, human_size
from ._paths import DatasetError, project_root
from ._sources import detect_datasets, load_sources, local_state
from .provenance import readers_of


def _card_for(key: str, store=None) -> Optional[Dict[str, Any]]:
    try:
        return get_dataset(key, store=store)
    except Exception:  # noqa: BLE001 - an unreachable index must not hide the list
        return None


def _scripts(records: List[dict]) -> List[dict]:
    """One row per script, newest run first."""
    best: Dict[str, dict] = {}
    for record in records:
        script = record["script"]
        current = best.get(script)
        if current is None or (record.get("started_at") or "") > (
            current.get("started_at") or ""
        ):
            best[script] = record
    return sorted(best.values(), key=lambda r: r.get("started_at") or "", reverse=True)


def project_datasets(
    root,
    *,
    store=None,
    record_store=None,
    providers: Optional[list] = None,
    cache=None,
    with_readers: bool = True,
) -> Dict[str, Any]:
    """List the public datasets of the project at ``root``.

    ``store`` is the catalogue index (for metadata of recorded keys);
    ``record_store`` the host's routing to the project's provenance store,
    passed to providers (contract 1.1). Returns::

        {"datasets": [view...], "provenance": {...}, "manifest_error": str|None}

    Each view carries the manifest fields plus ``recorded`` (bool),
    ``detected_at`` (local folder found on disk), ``local`` (verification of
    the local copy), ``cache`` (shared-cache state), ``fetch`` and
    ``scripts`` (who read it).
    """
    from ..cache import DatasetCache
    from ._fetch import job_status

    root = project_root(root)
    cache = cache or DatasetCache()
    manifest_error = None
    try:
        recorded = load_sources(root)["datasets"]
    except DatasetError as exc:
        recorded, manifest_error = [], str(exc)

    views: Dict[str, Dict[str, Any]] = {}
    for entry in recorded:
        views[entry["key"]] = {**entry, "recorded": True, "detected_at": None}

    for hit in detect_datasets(root):
        view = views.get(hit["key"])
        if view is None:
            card = _card_for(hit["key"], store) or {"key": hit["key"]}
            view = {
                **{k: card.get(k) for k in card if k not in ("snippet",)},
                "local_dir": hit["local_dir"],
                "recorded": False,
                "origin": "detected",
                "fetch": {"state": "on-disk"},
            }
            views[hit["key"]] = view
        view["detected_at"] = hit["local_dir"]
        view["provenance_file"] = hit.get("provenance_file")

    paths: List[str] = []
    sources: Dict[str, Dict[str, Any]] = {}
    for view in views.values():
        card = _card_for(view["key"], store)
        if card:
            for field in ("title", "repository_name", "doi_url", "citation", "snippet"):
                view.setdefault(field, card.get(field))
            if not view.get("snippet"):
                view["snippet"] = card.get("snippet")
        view["local"] = local_state(root, view)
        size = sum(int(f.get("size_bytes") or 0) for f in view.get("files") or [])
        view["size_bytes"] = size or None
        view["size"] = human_size(size) if size else None
        view["cache"] = (
            cache.status(view) if view.get("files") else {"state": "not-cached"}
        )
        fetch = view.get("fetch") or {}
        if fetch.get("job_id") and fetch.get("state") in ("queued", "running"):
            try:
                job = job_status(root, fetch["job_id"])
            except DatasetError:
                job = {"state": "interrupted"}
            view["fetch"] = {**fetch, "state": job["state"], "job": job}
        view_paths = [row["local_path"] for row in view["local"]["files"]]
        if view.get("provenance_file"):
            view_paths.append(view["provenance_file"])
        for row in view["local"]["files"]:
            sources[row["local_path"]] = {
                "key": view["key"],
                "repository": view.get("repository"),
                "accession": view.get("accession"),
                "version": view.get("version"),
                "doi": view.get("doi"),
                "sha256": row.get("sha256"),
            }
        view["_paths"] = view_paths
        paths.extend(view_paths)

    provenance = {"providers": [], "readers": {}, "sources": sources}
    if with_readers and paths:
        provenance = readers_of(
            root, paths, providers=providers, store=record_store, sources=sources
        )
    for view in views.values():
        records = [
            r for p in view.pop("_paths") for r in provenance["readers"].get(p, [])
        ]
        view["scripts"] = _scripts(records)

    ordered = sorted(views.values(), key=lambda v: (not v["recorded"], v["key"]))
    return {
        "datasets": ordered,
        "provenance": provenance,
        "manifest_error": manifest_error,
    }


__all__ = ["project_datasets"]

# EOF
