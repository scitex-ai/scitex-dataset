#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views/_curated.py

"""Curated AI-for-science datasets, and the shared cache (operators).

Masking boundary: these views describe a curated dataset — its source, the
derivation chain, the masking policies and COUNTS of what is held out. Names
of held-out and raw files are listed for operators only, and no view, for
anyone, returns the content of any file.
"""

from __future__ import annotations

try:
    from django.http import Http404, HttpResponse
    from django.views.decorators.http import require_GET, require_http_methods
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.views needs Django and scitex-sdk >= 0.3.26. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from ... import curated
from ._host import (
    dataset_cache,
    dataset_root,
    guarded_page,
    is_operator,
    page_access,
    render,
)


def _describe(request, benchmark):
    try:
        return curated.describe(
            benchmark, dataset_root(), operator=is_operator(request.user)
        )
    except KeyError:
        return None


@require_GET
@guarded_page
def curated_list(request):
    """The curated AI-for-science datasets prepared on this host."""
    return render(
        request,
        "scitex_dataset/curated_list.html",
        {
            "tab": "curated",
            "access": page_access(request),
            "curated": curated.list_curated(dataset_root()),
        },
    )


@require_GET
@guarded_page
def curated_detail(request, benchmark):
    """One curated dataset: source, derivation chain, what is masked."""
    access = page_access(request)
    described = _describe(request, benchmark)
    if described is None:
        raise Http404("curated dataset not found")
    return render(
        request,
        "scitex_dataset/curated_detail.html",
        {"tab": "curated", "access": access, "c": described},
    )




@require_http_methods(["GET", "POST"])
@guarded_page
def cache_page(request):
    """Shared-cache usage, entries and LRU eviction (operators only)."""
    if not is_operator(request.user):
        return HttpResponse("Operators only", status=403)
    cache = dataset_cache()
    result = None
    if request.method == "POST":
        result = cache.evict(dry_run=request.POST.get("dry_run") == "1")
    return render(
        request,
        "scitex_dataset/cache.html",
        {
            "tab": "cache",
            "access": None,
            "usage": cache.usage(),
            "entries": cache.entries(),
            "result": result,
        },
    )


# EOF
