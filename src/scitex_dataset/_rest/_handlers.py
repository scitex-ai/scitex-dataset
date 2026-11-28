#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_rest/_handlers.py

"""REST/MCP handlers: thin adapters over the Python API (:mod:`.._operations`).

Each takes the SDK's ``ApiCall`` (validated params; ``call.project`` already
authorized by the host for project-scoped operations), calls the matching
``scitex_dataset.<resource>_<verb>`` function and returns plain JSON data.
Authentication, scopes, rate limits, metering, idempotency and the error
envelope are the host's and the SDK's — never handled here. The only host
call is the storage gate before a fetch writes bytes.
"""

from __future__ import annotations

from typing import Any, Dict

from scitex_sdk.app.api_exec import ApiJob, ApiPage, ApiProblem

from .. import _operations as ops
from .._operations import DatasetNotFound, dataset_id
from ..cache import DatasetCache
from ..project import DatasetError, default_local_dir


def _setting(name: str):
    try:
        from django.conf import settings
    except ImportError:  # the server extra is absent: no host settings
        return None
    return getattr(settings, name, None) if settings.configured else None


def _index_store():
    value = _setting("SCITEX_DATASET_INDEX_STORE")
    if isinstance(value, str):
        from importlib import import_module

        module, _, attr = value.rpartition(".")
        value = getattr(import_module(module), attr)
    return value


def _cache() -> DatasetCache:
    return DatasetCache(_setting("SCITEX_DATASET_CACHE_DIR"))


def _public(card: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in card.items() if k != "citation_text"}
    out["id"] = dataset_id(card["key"])
    return out


def _not_found(exc) -> ApiProblem:
    return ApiProblem(404, "dataset.not_found", str(exc))


def _rejected(exc: DatasetError) -> ApiProblem:
    return ApiProblem(exc.status, "dataset.rejected", str(exc))


def _page(rows, call) -> ApiPage:
    page = rows[call.offset : call.offset + call.limit]
    return ApiPage(page, has_more=call.offset + len(page) < len(rows), total=len(rows))


def _username(call):
    return getattr(call.caller.user, "username", None) or None


def _record_store(call):
    """The host's routing to the project's record store (provenance contract 1.1)."""
    if call.request is None:
        return None
    from scitex_sdk.host import AccessError, CapabilityUnavailable, store_access

    try:
        store = store_access(call.request, call.project.id)
    except (AccessError, CapabilityUnavailable):
        return None
    return {
        "tenant_id": str(store.tenant_id),
        "dsn": store.dsn,
        "project_scope": store.project_scope,
        "owner_role": store.owner_role,
    }


# --- public datasets ----------------------------------------------------------


def search_datasets(call):
    p = call.params
    try:
        found = ops.datasets_list(
            p.get("q", ""),
            repository=p.get("repository"),
            domain=p.get("domain"),
            modality=p.get("modality"),
            license=p.get("license"),
            min_size_bytes=p.get("min_size_bytes"),
            max_size_bytes=p.get("max_size_bytes"),
            limit=call.limit,
            offset=call.offset,
            store=_index_store(),
        )
    except ValueError as exc:
        raise ApiProblem(400, "dataset.invalid_filter", str(exc)) from exc
    if found["index"]["status"] != "ok" and not found["results"]:
        raise ApiProblem(
            503,
            "dataset.index_unavailable",
            "the public-dataset index is not available on this host",
            retryable=True,
        )
    items = [_public(c) for c in found["results"]]
    return ApiPage(
        items, has_more=call.offset + len(items) < found["total"], total=found["total"]
    )


def get_dataset(call):
    try:
        return _public(ops.datasets_get(call.id, store=_index_store(), cache=_cache()))
    except DatasetNotFound as exc:
        raise _not_found(exc) from exc


def get_cache_entry(call):
    try:
        status = ops.cache_entries_get(call.id, store=_index_store(), cache=_cache())
    except DatasetNotFound as exc:
        raise _not_found(exc) from exc
    status.pop("path", None)  # a host path is not the caller's business
    return {"id": dataset_id(status["dataset"]), **status}


# --- a project's datasets -----------------------------------------------------


def _row(view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": dataset_id(view["key"]),
        "dataset": view["key"],
        "title": view.get("title"),
        "repository": view.get("repository"),
        "version": view.get("version"),
        "doi": view.get("doi"),
        "license": view.get("license"),
        "local_dir": view.get("local_dir"),
        "recorded": view["recorded"],
        "local_state": view["local"]["state"],
        "cache_state": view["cache"]["state"],
        "fetch_state": (view.get("fetch") or {}).get("state"),
        "size": view.get("size"),
        "size_bytes": view.get("size_bytes"),
        "files": [
            {k: f.get(k) for k in ("local_path", "sha256", "size_bytes", "state")}
            for f in view["local"]["files"]
        ],
        "scripts": [
            {
                k: r.get(k)
                for k in ("script", "run_id", "started_at", "status", "provider")
            }
            for r in view["scripts"]
        ],
    }


def list_project_datasets(call):
    views = ops.project_datasets_list(
        call.project.root,
        store=_index_store(),
        record_store=_record_store(call),
        cache=_cache(),
    )
    return _page([_row(v) for v in views], call)


def record_project_dataset(call):
    try:
        entry = ops.project_datasets_create(
            call.project.root,
            call.params["dataset"],
            call.params.get("local_dir"),
            added_by=_username(call),
            store=_index_store(),
        )
    except DatasetNotFound as exc:
        raise _not_found(exc) from exc
    except DatasetError as exc:
        raise _rejected(exc) from exc
    return {"id": dataset_id(entry["key"]), "dataset": entry["key"], **entry}


def begin_fetch(project, card, local_dir, *, username, request, local=False):
    """Admit the incoming bytes with the HOST, then start the fetch job.

    Shared by the REST operation and the app's form, so both ask the host's
    storage gate (quota, spend controls) the same way. ``local`` (a local MCP
    caller on its own machine) has no host gate. Without a gate a hosted call
    raises ``CapabilityUnavailable`` and nothing is written.
    """
    if not card.get("fetchable") or not card.get("files"):
        raise ApiProblem(
            422,
            "dataset.not_fetchable",
            f"{card.get('repository_name')} files cannot be fetched by scitex-dataset yet",
        )
    local_dir = local_dir or default_local_dir(card)
    from ..project._paths import inside

    target = inside(project.root, local_dir)
    incoming = [
        int(f.get("size_bytes") or 0)
        for f in card["files"]
        if not (target / f["path"]).is_file()
    ]
    if not local:
        from scitex_sdk.host import require_storage_write

        require_storage_write(request, project, incoming_sizes=incoming)
    try:
        return ops.project_datasets_fetch(
            project.root,
            card["key"],
            local_dir,
            added_by=username,
            store=_index_store(),
            cache=_cache(),
        )
    except DatasetError as exc:
        raise _rejected(exc) from exc


def fetch_project_dataset(call):
    try:
        card = ops.datasets_get(call.id, store=_index_store(), cache=_cache())
    except DatasetNotFound as exc:
        raise _not_found(exc) from exc
    job = begin_fetch(
        call.project,
        card,
        call.params.get("local_dir"),
        username=_username(call),
        request=call.request,
        local=call.caller.via == "local",
    )
    return ApiJob(
        {
            "id": job["id"],
            "state": job["state"],
            "percent": job["percent"],
            "message": job["message"],
            "links": {"self": f"/api/v1/dataset/jobs/{job['id']}"},
        }
    )


def get_job(call):
    try:
        job = ops.jobs_get(call.project.root, call.id)
    except DatasetError as exc:
        raise ApiProblem(404, "dataset.job_not_found", "no such job") from exc
    fields = (
        "id",
        "key",
        "state",
        "percent",
        "done_bytes",
        "total_bytes",
        "message",
        "local_dir",
        "started_at",
        "finished_at",
    )
    return {k: job.get(k) for k in fields}


# --- curated AI-for-science datasets -----------------------------------------


def _operator(call) -> bool:
    user = call.caller.user
    return bool(
        getattr(user, "is_staff", False) or getattr(user, "is_superuser", False)
    )


def list_curated(call):
    rows = [
        {"id": c["benchmark"], **c}
        for c in ops.curated_datasets_list(_setting("SCITEX_DATASET_ROOT"))
    ]
    return _page(rows, call)


def get_curated(call):
    """The masking boundary: hidden-role NAMES for operators only, content never."""
    try:
        described = ops.curated_datasets_get(
            call.id, _setting("SCITEX_DATASET_ROOT"), operator=_operator(call)
        )
    except DatasetNotFound as exc:
        raise _not_found(exc) from exc
    return {"id": call.id, **described}


# EOF
