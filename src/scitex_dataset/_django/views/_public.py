#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views/_public.py

"""Find public datasets: the search page, a dataset's page, and their JSON."""

from __future__ import annotations

from urllib.parse import urlencode

try:
    from django.http import Http404
    from django.urls import reverse
    from django.views.decorators.http import require_GET
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.views needs Django and scitex-sdk >= 0.3.26. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from ... import catalog
from ...project import default_local_dir, project_datasets
from ._host import (
    PROJECT_PARAM,
    budget,
    dataset_cache,
    error,
    guarded_page,
    index_store,
    page_access,
    render,
)

PAGE_SIZE = 20
_GB = 1_000**3
#: Size filter: choice -> (label, min_bytes, max_bytes).
SIZE_CHOICES = {
    "": ("Any size", None, None),
    "lt100mb": ("Under 100 MB", None, 100 * 1_000**2),
    "lt1gb": ("Under 1 GB", None, _GB),
    "lt10gb": ("Under 10 GB", None, 10 * _GB),
    "gt10gb": ("Over 10 GB", 10 * _GB, None),
}
_FILTERS = ("q", "repository", "domain", "modality", "license", "size")


def search_params(request) -> dict:
    size = request.GET.get("size", "")
    try:
        page = max(1, int(request.GET.get("page", "1")))
    except ValueError:
        page = 1
    return {
        "q": (request.GET.get("q") or "").strip()[:200],
        "repository": request.GET.get("repository") or "",
        "domain": request.GET.get("domain") or "",
        "modality": (request.GET.get("modality") or "").strip()[:40],
        "license": (request.GET.get("license") or "").strip()[:200],
        "size": size if size in SIZE_CHOICES else "",
        "page": page,
    }


def run_search(params: dict) -> dict:
    _, min_bytes, max_bytes = SIZE_CHOICES[params["size"]]
    try:
        return catalog.search_public(
            params["q"],
            repository=params["repository"] or None,
            domain=params["domain"] or None,
            modality=params["modality"] or None,
            license=params["license"] or None,
            min_size_bytes=min_bytes,
            max_size_bytes=max_bytes,
            limit=PAGE_SIZE,
            offset=(params["page"] - 1) * PAGE_SIZE,
            store=index_store(),
        )
    except ValueError as exc:
        return {"results": [], "total": 0, "error": str(exc), "index": {}}


def with_cache(cards, cache):
    for card in cards:
        card["cache"] = (
            cache.status(card) if card.get("files") else {"state": "not-cached"}
        )
    return cards


def card_or_404(key: str) -> dict:
    try:
        card = catalog.get_dataset(key, store=index_store())
    except ValueError as exc:
        raise Http404("dataset not found") from exc
    if card is None:
        raise Http404("dataset not found")
    return card


@require_GET
@guarded_page
def index(request):
    """Project selected -> its public datasets; else the search page."""
    if request.GET.get(PROJECT_PARAM):
        from ._project import project_page

        return project_page.__wrapped__(request)
    return search.__wrapped__(request)


@require_GET
@guarded_page
def search(request):
    """Search public datasets across every supported repository."""
    access = page_access(request)
    params = search_params(request)
    searched = any(params[k] for k in _FILTERS)
    found = run_search(params) if searched else None
    if found:
        with_cache(found["results"], dataset_cache())
    facets = (found or {}).get("index", {}).get("facets") or {}
    total = (found or {}).get("total", 0)
    query = {k: v for k, v in params.items() if v and k != "page"}
    if access:
        query[PROJECT_PARAM] = access.id
    repositories = catalog.repositories()
    return render(
        request,
        "scitex_dataset/search.html",
        {
            "tab": "search",
            "access": access,
            "params": params,
            "found": found,
            "repositories": repositories,
            "domains": sorted({r["domain"] for r in repositories}),
            "licenses": sorted(
                (k for k in (facets.get("license") or {}) if k), key=str.lower
            ),
            "size_choices": [(k, v[0]) for k, v in SIZE_CHOICES.items()],
            "page": params["page"],
            "has_next": params["page"] * PAGE_SIZE < total,
            "next_query": urlencode({**query, "page": params["page"] + 1}),
            "prev_query": urlencode({**query, "page": params["page"] - 1}),
        },
    )


@require_GET
@guarded_page
def detail(request, key):
    """One public dataset: metadata, files, licence, citation, snippet, use."""
    access = page_access(request)
    try:
        card = card_or_404(key)
    except Http404:
        raise
    except Exception as exc:  # noqa: BLE001 - the index is a remote store
        return render(
            request,
            "scitex_dataset/unavailable.html",
            {
                "tab": "search",
                "access": access,
                "problem": f"{type(exc).__name__}: {exc}",
            },
            status=503,
        )
    cache = dataset_cache()
    card["cache"] = cache.status(card) if card.get("files") else {"state": "not-cached"}
    used = None
    report = {"reported": False}
    if access:
        listing = project_datasets(
            access.root, store=index_store(), cache=cache, with_readers=False
        )
        used = next((d for d in listing["datasets"] if d["key"] == card["key"]), None)
        if access.can_write:
            report = budget(request, access)
    return render(
        request,
        "scitex_dataset/detail.html",
        {
            "tab": "search",
            "access": access,
            "card": card,
            "used": used,
            "budget": report,
            "local_dir": (used or {}).get("local_dir") or default_local_dir(card),
            "project_url": (
                reverse("scitex_dataset:index")
                + "?"
                + urlencode({PROJECT_PARAM: access.id})
                if access
                else ""
            ),
        },
    )






def legacy_redirect(request):
    """The 0.15 JSON endpoints are gone: 410 with where to go instead.

    Upload and path registration moved to the host's file browser; the
    project's public datasets are ``GET /api/v1/dataset/project-datasets``.
    """
    return error(
        410,
        "this endpoint was removed: programs use /api/v1/dataset/ (for this "
        "project's datasets: GET /api/v1/dataset/project-datasets?project=owner/slug); "
        "uploads moved to the Hub's file browser",
        search_url="/api/v1/dataset/datasets",
    )


# EOF
