#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views.py

"""Views of the Dataset app — thin: authorize through the host, call the logic.

Authorization is the HOST's, reached only through
``scitex_sdk.host.project_access`` (the Hub implements it over its project
model): a project member may read, a member with write permission (a
maintainer) may write, and anyone else gets 404 — the same answer as a
project that does not exist, so a non-member cannot learn that it does.

HTML forms post to the same JSON endpoints with ``_redirect=1`` and are sent
back to the page (303) with a one-line status message; API clients get JSON.
"""

from __future__ import annotations

from urllib.parse import urlencode

try:
    from django.http import Http404, HttpResponse, HttpResponseRedirect, JsonResponse
    from django.template.loader import render_to_string
    from django.urls import reverse
    from django.views.decorators.http import require_GET, require_POST
    from scitex_sdk.host import AccessError, CapabilityUnavailable, project_access
    from scitex_sdk.ui.branding import shell_context
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.views needs Django and scitex-sdk. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from ..project import DatasetError, list_datasets, register_dataset, save_upload

PROJECT_PARAM = "project"


def _access(request, *, write: bool, remember: bool = False):
    """The authorized project; :class:`AccessError` carries the refusal status.

    Only page navigation remembers the selected project; resource requests
    (API calls) never move it (the SDK's ``remember`` rule).
    """
    return project_access(request, write=write, remember=remember)


def _error(status: int, message: str) -> JsonResponse:
    return JsonResponse({"success": False, "error": message}, status=status)


def _refusal(exc: Exception) -> JsonResponse:
    if isinstance(exc, CapabilityUnavailable):
        return _error(503, "project access is unavailable on this host")
    status = getattr(exc, "status", 400)
    # One answer for "absent" and "not yours": never confirm a project exists.
    message = "project not found" if status == 404 else str(exc)
    return _error(status, message)


def _user_label(request) -> str | None:
    user = getattr(request, "user", None)
    return getattr(user, "username", None) or None


def _back(request, access, message: str, ok: bool) -> HttpResponseRedirect:
    query = urlencode({PROJECT_PARAM: access.id, "ok" if ok else "error": message})
    return HttpResponseRedirect(
        f"{reverse('scitex_dataset:index')}?{query}", status=303
    )


def _wants_redirect(request) -> bool:
    return request.POST.get("_redirect") == "1"


@require_GET
def index(request):
    """The Dataset page for the selected project (``?project=owner/slug``)."""
    from .. import __version__

    context = shell_context("Dataset")
    context.update(
        dataset_version=__version__,
        ok=request.GET.get("ok", ""),
        error=request.GET.get("error", ""),
    )
    try:
        access = _access(request, write=False, remember=True)
    except AccessError as exc:
        if exc.status == 404 and request.GET.get(PROJECT_PARAM):
            raise Http404("project not found") from exc
        if exc.status != 404:
            return HttpResponse(str(exc), status=exc.status)
        # No project selected yet: the page offers the host's project picker.
        context.update(dataset_project_id="", project=None)
        return HttpResponse(
            render_to_string("scitex_dataset/index.html", context, request=request)
        )
    except CapabilityUnavailable as exc:
        return HttpResponse(
            f"Project access is unavailable on this host: {exc}", status=503
        )

    try:
        listing = list_datasets(access.root)
    except DatasetError as exc:
        raise Http404("project workspace not found") from exc
    readers = (listing.get("provenance") or {}).get("readers", {})
    for row in listing["files"] + listing["registered"]:
        row["readers"] = readers.get(row.get("path") or row.get("location") or "", [])
    context.update(
        dataset_project_id=access.id,
        project={"id": access.id, "name": access.name, "can_write": access.can_write},
        listing=listing,
        action_query=urlencode({PROJECT_PARAM: access.id}),
    )
    return HttpResponse(
        render_to_string("scitex_dataset/index.html", context, request=request)
    )


@require_GET
def datasets(request):
    """JSON listing of the selected project's datasets (members)."""
    try:
        access = _access(request, write=False)
    except (AccessError, CapabilityUnavailable) as exc:
        return _refusal(exc)
    try:
        listing = list_datasets(access.root)
    except DatasetError as exc:
        return _error(exc.status, str(exc))
    return JsonResponse(
        {
            "success": True,
            "project": {
                "id": access.id,
                "name": access.name,
                "can_write": access.can_write,
            },
            **listing,
        }
    )


@require_POST
def upload(request):
    """Upload one file into ``data/[subdir/]`` (maintainers)."""
    try:
        access = _access(request, write=True)
    except (AccessError, CapabilityUnavailable) as exc:
        return _refusal(exc)
    uploaded = request.FILES.get("file")
    try:
        if uploaded is None:
            raise DatasetError("choose a file to upload")
        saved = save_upload(
            access.root,
            uploaded.name,
            uploaded.chunks(),
            subdir=request.POST.get("subdir", ""),
        )
    except DatasetError as exc:
        if _wants_redirect(request):
            return _back(request, access, str(exc), ok=False)
        return _error(exc.status, str(exc))
    if _wants_redirect(request):
        return _back(request, access, f"Uploaded {saved['path']}", ok=True)
    return JsonResponse({"success": True, "file": saved}, status=201)


@require_POST
def register(request):
    """Register a project ``path`` or an external ``url`` (maintainers)."""
    try:
        access = _access(request, write=True)
    except (AccessError, CapabilityUnavailable) as exc:
        return _refusal(exc)
    path = (request.POST.get("path") or "").strip()
    url = (request.POST.get("url") or "").strip()
    try:
        if bool(path) == bool(url):
            raise DatasetError("give exactly one of 'path' or 'url'")
        entry = register_dataset(
            access.root,
            path or url,
            kind="path" if path else "url",
            name=request.POST.get("name", ""),
            note=request.POST.get("note", ""),
            registered_by=_user_label(request),
        )
    except DatasetError as exc:
        if _wants_redirect(request):
            return _back(request, access, str(exc), ok=False)
        return _error(exc.status, str(exc))
    if _wants_redirect(request):
        return _back(request, access, f"Registered {entry['name']}", ok=True)
    return JsonResponse({"success": True, "dataset": entry}, status=201)


# EOF
