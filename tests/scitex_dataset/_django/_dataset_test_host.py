#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/_dataset_test_host.py

"""A minimal SciTeX HOST for the Dataset app's tests — a real implementation.

It implements the SDK host contract the Hub implements:

* ``SCITEX_PROJECT_PROVIDER`` / ``SCITEX_PROJECT_STORAGE`` over an in-memory
  role table, with the Hub's project rules: members read, members with write
  permission write, anyone else cannot see the project at all;
* ``SCITEX_STORAGE_GATE`` (scitex-sdk >= 0.3.26) with a byte quota and a
  spend switch, the two refusals the Hub's spend controls produce.

Identity comes from the ``X-Test-User`` header through
:class:`HeaderUserMiddleware`, standing in for the host's session login;
names in :data:`STAFF` are operators.
"""

from __future__ import annotations

from types import SimpleNamespace

from scitex_sdk.app.api_exec import ApiCaller
from scitex_sdk.app.api_serve import InMemoryIdempotencyStore
from scitex_sdk.host import StorageBudget, StorageRefused
from scitex_sdk.ui.project_scope import ProjectEntry

#: ``{project_id: {"root": Path, "roles": {username: "read" | "write"}}}``.
PROJECTS: dict = {}
#: Operators (``is_staff``).
STAFF: set = set()
#: The storage gate's state: bytes used, quota, and whether spend is blocked.
GATE: dict = {}
_LAST: dict = {}
#: Personal access tokens: ``{token: (username, scopes)}``.
TOKENS: dict = {}
IDEMPOTENCY = InMemoryIdempotencyStore()


def _username(request):
    user = getattr(request, "user", None)
    if not getattr(user, "is_authenticated", False):
        return None
    return user.username


def _role(project_id, request):
    project = PROJECTS.get(project_id)
    name = _username(request)
    if project is None or name is None:
        return None
    return project["roles"].get(name)


class RoleTableProjectProvider:
    def list_projects(self, request):
        return [
            ProjectEntry(
                id=pid, name=pid.rpartition("/")[2], detail=pid.partition("/")[0]
            )
            for pid in sorted(PROJECTS)
            if _role(pid, request)
        ]

    def last_visited(self, request):
        return _LAST.get(_username(request))

    def remember(self, request, project_id):
        if _role(project_id, request):
            _LAST[_username(request)] = project_id


class RoleTableProjectStorage:
    def project_path(self, project_id, request):
        if not _role(project_id, request):
            return None
        return str(PROJECTS[project_id]["root"])

    def can_write(self, project_id, request):
        return _role(project_id, request) == "write"


class QuotaStorageGate:
    """Byte quota + spend switch; records every admission it was asked for."""

    def budget(self, project_id, request):
        return StorageBudget(
            used_bytes=GATE.get("used", 0),
            quota_bytes=GATE.get("quota"),
            spend_level="blocked" if GATE.get("blocked") else "ok",
        )

    def require_storage_write(self, project_id, request, *, incoming_sizes):
        GATE.setdefault("asked", []).append((project_id, list(incoming_sizes)))
        if GATE.get("blocked"):
            raise StorageRefused("Spend cap reached", 402, code="spend-cap")
        quota = GATE.get("quota")
        if quota is not None and GATE.get("used", 0) + sum(incoming_sizes) > quota:
            raise StorageRefused("Storage quota exceeded", 413, code="quota")


class HeaderUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        name = request.headers.get("X-Test-User")
        request.user = SimpleNamespace(
            username=name or "",
            is_authenticated=bool(name),
            is_staff=name in STAFF,
            is_superuser=False,
            pk=name,
        )
        return self.get_response(request)


def api_caller(request):
    """The host's API authenticator: a bearer token, else the session user.

    A session user (the app's own pages) holds every scope; a token holds the
    scopes it was minted with. Either way the user's PROJECT rights still come
    from the role table through ``project_access``.
    """
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        found = TOKENS.get(header[7:].strip())
        if found is None:
            return None
        name, scopes = found
        user = SimpleNamespace(
            username=name, is_authenticated=True, is_staff=name in STAFF,
            is_superuser=False, pk=name,
        )
        request.user = user
        return ApiCaller(user=user, scopes=tuple(scopes), via="token", token_id=header[7:12])
    user = getattr(request, "user", None)
    if not getattr(user, "is_authenticated", False):
        return None
    return ApiCaller(user=user, scopes=("*",), via="session")


def reset():
    global IDEMPOTENCY
    IDEMPOTENCY = InMemoryIdempotencyStore()
    TOKENS.clear()
    PROJECTS.clear()
    STAFF.clear()
    GATE.clear()
    _LAST.clear()


# EOF
