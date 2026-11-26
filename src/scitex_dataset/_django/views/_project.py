#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views/_project.py

"""Use a public dataset in a project: record it, fetch it, list what is used."""

from __future__ import annotations

from urllib.parse import urlencode

try:
    from django.http import HttpResponseRedirect
    from django.urls import reverse
    from django.views.decorators.http import require_GET, require_POST
    from scitex_sdk.app.api_exec import ApiProblem
    from scitex_sdk.host import CapabilityUnavailable, StorageRefused
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.views needs Django and scitex-sdk >= 0.3.26. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from ..._rest._handlers import begin_fetch
from ...project import DatasetError, default_local_dir, project_datasets, record_dataset
from ._host import (
    PROJECT_PARAM,
    api_access,
    budget,
    dataset_cache,
    guarded_page,
    index_store,
    page_access,
    record_store,
    render,
)
from ._public import card_or_404


@require_GET
@guarded_page
def project_page(request):
    """The public datasets this project uses, with fetch status and readers."""
    access = page_access(request)
    if access is None:
        return HttpResponseRedirect(reverse("scitex_dataset:search"))
    listing = project_datasets(
        access.root,
        store=index_store(),
        record_store=record_store(request, access),
        cache=dataset_cache(),
    )
    return render(
        request,
        "scitex_dataset/project.html",
        {
            "tab": "project",
            "access": access,
            "listing": listing,
            "budget": budget(request, access)
            if access.can_write
            else {"reported": False},
        },
    )


def _back_to(access, key: str, message: str, ok: bool):
    query = urlencode({PROJECT_PARAM: access.id, "ok" if ok else "error": message})
    return HttpResponseRedirect(
        f"{reverse('scitex_dataset:detail', args=[key])}?{query}", status=303
    )


@require_POST
@guarded_page
def use_in_project(request):
    """The page's form: record a public dataset, or record and fetch it.

    Write permission required. A fetch is admitted by the host's storage gate
    (see :func:`scitex_dataset._rest._handlers.begin_fetch`); the page then
    follows the job through the REST API. Programs use the REST API directly.
    """
    access = api_access(request, write=True)
    key = (request.POST.get("key") or "").strip()
    card = card_or_404(key)
    local_dir = (request.POST.get("local_dir") or "").strip() or default_local_dir(card)
    by = getattr(request.user, "username", None) or None
    try:
        if request.POST.get("fetch") == "1":
            job = begin_fetch(access, card, local_dir, username=by, request=request)
            return _back_to(
                access, key, f"Fetching into {local_dir} in the background (job {job['id'][:8]})", ok=True
            )
        record_dataset(access.root, card, local_dir=local_dir, added_by=by)
    except (DatasetError, ApiProblem, StorageRefused, CapabilityUnavailable) as exc:
        return _back_to(access, key, _refusal_text(exc), ok=False)
    return _back_to(access, key, f"Recorded in {access.name}", ok=True)


def _refusal_text(exc) -> str:
    if isinstance(exc, CapabilityUnavailable):
        return (
            "this host does not report storage quotas, so fetching into project "
            "storage is disabled; record the dataset and use the code snippet"
        )
    return getattr(exc, "detail", None) or str(exc)


# EOF
