#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/_fetch.py

"""Fetch a recorded public dataset into a project, as a background job.

The bytes come from the shared cache (:mod:`scitex_dataset.cache`): a dataset
already cached by any project or user is not downloaded again. The project
then gets HARD LINKS to the cached files in its ``local_dir`` — no second copy
on disk, and evicting the cache never breaks the project. Only when the cache
and the project are on different filesystems are the files copied; the
manifest records which (``fetch.mode``: ``hardlink`` | ``copy``).

Admission (quota, spend controls) is the HOST's and happens before
:func:`start_fetch` is called — see the Django views, which ask the host
through ``scitex_sdk.host.require_storage_write``. This module never decides
whether a write is affordable.

A job is a JSON file at ``<root>/.scitex/dataset/jobs/<id>.json`` with
``state`` (``queued`` → ``running`` → ``done`` | ``failed``), ``done_bytes``,
``total_bytes``, ``percent`` and ``message``, updated as bytes arrive, and a
``heartbeat``. It runs in a daemon thread of the calling process; a job whose
heartbeat stops (the process ended) reads as ``interrupted``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ._paths import DatasetError, inside, project_root, same_device, sha256_of
from ._sources import record_dataset, update_fetch

JOBS_RELPATH = ".scitex/dataset/jobs"
#: A running job whose heartbeat is older than this reads as interrupted.
HEARTBEAT_STALE_SECONDS = 120

_LIVE: Dict[str, threading.Thread] = {}
_LIVE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_path(root: Path, job_id: str) -> Path:
    if not job_id or not all(c in "0123456789abcdef" for c in job_id):
        raise DatasetError("unknown job", 404)
    return root / JOBS_RELPATH / f"{job_id}.json"


def _write_job(path: Path, job: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".job-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(job, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def job_status(root, job_id: str) -> Dict[str, Any]:
    """The job's state; ``interrupted`` when its worker is gone."""
    root = project_root(root)
    path = _job_path(root, job_id)
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError("unknown job", 404) from exc
    if job.get("state") in ("queued", "running"):
        with _LIVE_LOCK:
            thread = _LIVE.get(job_id)
        alive = thread is not None and thread.is_alive()
        age = time.time() - float(job.get("heartbeat_epoch") or 0)
        if not alive and age > HEARTBEAT_STALE_SECONDS:
            job["state"] = "interrupted"
            job["message"] = "the fetch stopped (the server restarted); start it again"
    return job


def _link_into_project(
    root: Path, entry: dict, cache_files: Path, local_dir: str
) -> Dict[str, Any]:
    """Hard-link (or copy across filesystems) the cached files into the project."""
    target_dir = inside(root, local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    mode = "hardlink" if same_device(cache_files, target_dir) else "copy"
    linked = []
    for item in entry["files"]:
        source = cache_files / item["path"]
        target = (target_dir / item["path"]).resolve()
        if not target.is_relative_to(target_dir.resolve()):
            raise DatasetError(f"file path escapes the dataset folder: {item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_of(target) != item["sha256"]:
                raise DatasetError(
                    f"{target.relative_to(root)} already exists with different "
                    "content; it was left untouched",
                    409,
                )
        elif mode == "hardlink":
            os.link(source, target)
        else:
            tmp = target.with_name(f".{target.name}.part")
            shutil.copy2(source, tmp)
            if sha256_of(tmp) != item["sha256"]:
                tmp.unlink(missing_ok=True)
                raise DatasetError(f"copy of {item['path']} did not verify")
            os.replace(tmp, target)
        linked.append(dict(item))
    return {"mode": mode, "files": linked}


def _run(root: Path, job_id: str, card: dict, local_dir: str, cache) -> None:
    path = _job_path(root, job_id)
    job = json.loads(path.read_text(encoding="utf-8"))
    last = [0.0]

    def progress(done: int, total: Optional[int], message: str) -> None:
        now = time.time()
        if now - last[0] < 0.5 and done != total:
            return
        last[0] = now
        job.update(
            state="running",
            done_bytes=done,
            total_bytes=total,
            percent=None if not total else round(100.0 * done / total, 1),
            message=message,
            heartbeat_epoch=now,
        )
        _write_job(path, job)

    try:
        progress(0, card.get("size_bytes"), "checking the shared cache")
        entry = cache.fetch(card, progress=progress)
        files_dir = cache.files_dir(card["key"], card.get("version"))
        result = _link_into_project(root, entry, files_dir, local_dir)
        update_fetch(
            root,
            card["key"],
            state="fetched",
            job_id=job_id,
            mode=result["mode"],
            fetched_at=_now(),
            error=None,
            files=result["files"],
        )
        job.update(
            state="done",
            percent=100.0,
            message=f"{len(result['files'])} files in {local_dir} ({result['mode']})",
            heartbeat_epoch=time.time(),
            finished_at=_now(),
        )
    except Exception as exc:  # noqa: BLE001 - reported to the user, never swallowed
        update_fetch(root, card["key"], state="failed", job_id=job_id, error=str(exc))
        job.update(
            state="failed",
            message=str(exc),
            heartbeat_epoch=time.time(),
            finished_at=_now(),
        )
    _write_job(path, job)


def start_fetch(
    root,
    card: Dict[str, Any],
    *,
    local_dir: Optional[str] = None,
    added_by: Optional[str] = None,
    cache=None,
    wait: bool = False,
) -> Dict[str, Any]:
    """Record ``card`` in the project and fetch it in a background thread.

    Returns the queued job. ``wait=True`` runs it to completion first (for
    scripts and tests). Raises :class:`DatasetError` 409 when a fetch of the
    same dataset is already running in this project.
    """
    from ..cache import DatasetCache

    root = project_root(root)
    if not card.get("fetchable"):
        raise DatasetError(
            f"{card.get('repository_name')} datasets cannot be fetched by "
            "scitex-dataset yet; record it and use the snippet instead",
            422,
        )
    entry = record_dataset(root, card, local_dir=local_dir, added_by=added_by)
    current = entry.get("fetch") or {}
    if current.get("state") in ("queued", "running") and current.get("job_id"):
        if job_status(root, current["job_id"])["state"] in ("queued", "running"):
            raise DatasetError("this dataset is already being fetched", 409)
    cache = cache or DatasetCache()
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "key": card["key"],
        "title": card.get("title"),
        "local_dir": entry["local_dir"],
        "state": "queued",
        "done_bytes": 0,
        "total_bytes": card.get("size_bytes"),
        "percent": 0.0,
        "message": "queued",
        "started_by": added_by,
        "started_at": _now(),
        "heartbeat_epoch": time.time(),
    }
    _write_job(_job_path(root, job_id), job)
    update_fetch(root, card["key"], state="queued", job_id=job_id, error=None)
    thread = threading.Thread(
        target=_run,
        args=(root, job_id, card, entry["local_dir"], cache),
        name=f"scitex-dataset-fetch-{job_id[:8]}",
        daemon=True,
    )
    with _LIVE_LOCK:
        _LIVE[job_id] = thread
    thread.start()
    if wait:
        thread.join()
        return job_status(root, job_id)
    return job


__all__ = ["JOBS_RELPATH", "job_status", "start_fetch"]

# EOF
