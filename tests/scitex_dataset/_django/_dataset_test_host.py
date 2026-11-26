#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/_dataset_test_host.py

"""A minimal SciTeX HOST for the Dataset app's tests — a real implementation.

It implements the SDK host contract the Hub implements (``ProjectProvider`` for
``SCITEX_PROJECT_PROVIDER`` and the storage capability for
``SCITEX_PROJECT_STORAGE``) over an in-memory role table, with the Hub's
project model rules: members can read, members with write permission
(maintainers) can write, anyone else cannot see the project at all.

Identity comes from the ``X-Test-User`` header through
:class:`HeaderUserMiddleware`, standing in for the host's session login.
"""

from __future__ import annotations

from types import SimpleNamespace

from scitex_sdk.ui.project_scope import ProjectEntry

#: ``{project_id: {"root": Path, "roles": {username: "read" | "write"}}}``.
PROJECTS: dict = {}
_LAST: dict = {}


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


class HeaderUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        name = request.headers.get("X-Test-User")
        request.user = SimpleNamespace(
            username=name or "", is_authenticated=bool(name), pk=name
        )
        return self.get_response(request)


def reset():
    PROJECTS.clear()
    _LAST.clear()


# EOF
