#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views/_host.py

"""Host plumbing shared by the views: identity, project access, refusals.

Authorization is the HOST's, reached only through ``scitex_sdk.host``:

* every page and API needs a logged-in user;
* naming a project (``?project=owner/slug``) needs membership — a non-member
  gets 404, the same answer as a project that does not exist;
* writes need the host's write permission (403 otherwise) and, for bytes,
  the host's storage gate (quota and spend controls);
* operators (staff) see curated datasets' hidden-role inventories and manage
  the shared cache.

Host settings this app reads (all optional): ``SCITEX_DATASET_INDEX_STORE``
(a ``scitex_dev.store.Store``, or a dotted path to a callable returning one,
for the catalogue index), ``SCITEX_DATASET_CACHE_DIR`` (the shared cache) and
``SCITEX_DATASET_ROOT`` (where curated benchmarks are prepared).
"""

from __future__ import annotations

from functools import wraps
from typing import Optional
from urllib.parse import urlencode

try:
    from django.conf import settings
    from django.http import Http404, HttpResponse, HttpResponseRedirect, JsonResponse
    from django.template.loader import render_to_string
    from django.utils.module_loading import import_string
    from scitex_sdk.host import (
        AccessError,
        CapabilityUnavailable,
        StorageRefused,
        authenticated_user,
        project_access,
        storage_budget,
        store_access,
    )
    from scitex_sdk.ui.branding import shell_context
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.views needs Django and scitex-sdk >= 0.3.26. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from ...cache import DatasetCache
from ...catalog import human_size
from ...project import DatasetError

PROJECT_PARAM = "project"


def index_store():
    """The catalogue index as a LAZY factory (opened only when needed), or None."""
    value = getattr(settings, "SCITEX_DATASET_INDEX_STORE", None)
    if value is None:
        return None
    if isinstance(value, str):
        value = import_string(value)
    if hasattr(value, "search"):
        return value
    return value


def dataset_cache() -> DatasetCache:
    return DatasetCache(getattr(settings, "SCITEX_DATASET_CACHE_DIR", None))


def dataset_root():
    return getattr(settings, "SCITEX_DATASET_ROOT", None)


def is_operator(user) -> bool:
    return bool(
        getattr(user, "is_staff", False) or getattr(user, "is_superuser", False)
    )


def _user(request):
    user = authenticated_user(request)
    if not getattr(user, "is_authenticated", False):
        return None
    request.user = user
    return user


def error(status: int, message: str, **extra) -> JsonResponse:
    return JsonResponse({"success": False, "error": message, **extra}, status=status)


def refusal(exc: Exception) -> JsonResponse:
    if isinstance(exc, StorageRefused):
        return error(exc.status, str(exc), code=exc.code, link=exc.link)
    if isinstance(exc, CapabilityUnavailable):
        return error(503, "project access is unavailable on this host")
    status = getattr(exc, "status", 400)
    # One answer for "absent" and "not yours": never confirm a project exists.
    message = "project not found" if status == 404 else str(exc)
    return error(status, message)


def _login_redirect(request) -> HttpResponseRedirect:
    login = getattr(settings, "LOGIN_URL", "/accounts/login/")
    return HttpResponseRedirect(
        f"{login}?{urlencode({'next': request.get_full_path()})}"
    )


def page_access(request):
    """The project named by ``?project=`` (members only), or ``None``.

    A non-member or an unknown project raises :class:`Http404`, so the page
    never confirms that a project exists.
    """
    if not request.GET.get(PROJECT_PARAM):
        return None
    try:
        return project_access(request, write=False, remember=True)
    except AccessError as exc:
        if exc.status == 401:
            raise
        raise Http404("project not found") from exc


def api_access(request, *, write: bool):
    return project_access(request, write=write, remember=False)


def render(request, template: str, context: dict, status: int = 200) -> HttpResponse:
    from ... import __version__

    base = shell_context("Dataset")
    base.update(dataset_version=__version__, ok=request.GET.get("ok", ""))
    base.update(error=request.GET.get("error", ""))
    base.update(context)
    access = context.get("access")
    base.setdefault("dataset_project_id", access.id if access else "")
    base.setdefault(
        "project_query", urlencode({PROJECT_PARAM: access.id}) if access else ""
    )
    base.setdefault("is_operator", is_operator(getattr(request, "user", None)))
    return HttpResponse(
        render_to_string(template, base, request=request), status=status
    )


def guarded_page(view):
    """Login required; unknown/non-member project -> 404; host down -> 503."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if _user(request) is None:
            return _login_redirect(request)
        try:
            return view(request, *args, **kwargs)
        except CapabilityUnavailable as exc:
            return HttpResponse(
                f"Project access is unavailable on this host: {exc}", status=503
            )
        except AccessError as exc:
            if exc.status == 401:
                return _login_redirect(request)
            raise Http404("project not found") from exc

    return wrapper


def guarded_api(view):
    """Login required (401); refusals as JSON with the host's status."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if _user(request) is None:
            return error(401, "authentication required")
        try:
            return view(request, *args, **kwargs)
        except (AccessError, CapabilityUnavailable) as exc:
            return refusal(exc)
        except DatasetError as exc:
            return error(exc.status, str(exc))

    return wrapper


def record_store(request, access) -> Optional[dict]:
    """The host's routing to the project's record store, when it keeps one.

    A host without per-project stores answers ``CapabilityUnavailable``; the
    providers then read the store they are configured with, which on such a
    host IS the project's store.
    """
    try:
        store = store_access(request, access.id)
    except (CapabilityUnavailable, AccessError):
        return None
    return {
        "tenant_id": str(store.tenant_id),
        "dsn": store.dsn,
        "project_scope": store.project_scope,
        "owner_role": store.owner_role,
    }


def budget(request, access) -> dict:
    """The host's storage/spend report, or ``{"reported": False}``."""
    try:
        reported = storage_budget(request, access)
    except CapabilityUnavailable:
        return {"reported": False}
    return {
        "reported": True,
        "used": human_size(reported.used_bytes),
        "quota": human_size(reported.quota_bytes),
        "remaining": human_size(reported.remaining_bytes),
        "remaining_bytes": reported.remaining_bytes,
        "spend_level": reported.spend_level,
        "spend_percent": reported.spend_percent,
        "link": reported.link,
    }


# EOF
