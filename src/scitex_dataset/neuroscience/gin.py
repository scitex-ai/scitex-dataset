#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/neuroscience/gin.py

"""
GIN (G-Node Infrastructure) dataset adapter.

GIN — `<https://gin.g-node.org/>`_ — hosts a growing catalog of
neuroscience datasets behind a Gogs-style REST API and a `git-annex`
backed object store. The web frontend exposes annexed blobs at
``/{owner}/{repo}/raw/{branch}/{path}`` which resolves the annex
pointer server-side and streams the bytes back — no client-side
``git-annex`` or ``gin-cli`` installation required for the primary
download path.

This adapter ships three functions in the per-source convention:

- :func:`gin_search` — query the repos API (``/api/v1/repos/search``)
- :func:`gin_info`   — fetch metadata for one repo (``/api/v1/repos/{owner}/{repo}``)
- :func:`gin_download` — git clone + per-file HTTPS GET; idempotent;
  resumable; per-file byte logging

Optional `datalad` mode is wired through
:func:`scitex_dev.try_import_optional`; if `datalad` (and its
`git-annex` backend) is installed, the adapter delegates to
``datalad.api.get`` for resumable, parallel annex retrieval. Otherwise
the pure-HTTPS path is used. No silent fallback: the install hint is
surfaced via :func:`scitex_dev.last_install_hint`.

References
----------
- GIN API:           https://gin.g-node.org/api/swagger
- ripple-wm dogfood: https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Iterable, Optional

import httpx as _httpx  # noqa: N812
# Keep exception classes pinned to the real httpx module so tests can
# swap the module-level ``_httpx`` reference for a stub without
# breaking exception handling.
from httpx import HTTPError as _HTTPError

try:
    from scitex_dev import supports_return_as, try_import_optional
except ImportError:  # pragma: no cover — only when scitex_dev is absent
    def supports_return_as(fn):  # type: ignore[misc]
        return fn

    def try_import_optional(*_a, **_k):  # type: ignore[misc]
        return None

from .._config import runtime_dir as _runtime_dir, user_root as _user_root

# ----------------------------- constants ----------------------------- #

GIN_HOST = "https://gin.g-node.org"
GIN_API_V1 = f"{GIN_HOST}/api/v1"

#: Annex pointer files look like ``/annex/objects/MD5-s410499073--<hex>``.
#: We use the ``s<bytes>`` prefix as ground-truth declared file size.
_ANNEX_POINTER_RE = re.compile(
    r"^/annex/objects/(?P<backend>[A-Z0-9]+)"
    r"-s(?P<size>\d+)--(?P<key>[0-9a-f]+)$"
)

__all__ = [
    "GIN_HOST",
    "GIN_API_V1",
    "gin_search",
    "gin_info",
    "gin_download",
    "gin_format",
    "fetch_all_datasets",
    "format_dataset",
]


# ----------------------------- helpers ------------------------------- #

def _split_repo_id(repo_id: str) -> tuple[str, str]:
    """Split ``owner/repo`` (no leading slash, no ``.git`` suffix)."""
    rid = repo_id.strip("/")
    if rid.endswith(".git"):
        rid = rid[:-4]
    if rid.count("/") != 1:
        raise ValueError(
            f"GIN repo_id must be 'owner/repo' (got {repo_id!r})"
        )
    owner, repo = rid.split("/", 1)
    if not owner or not repo:
        raise ValueError(f"empty owner or repo in {repo_id!r}")
    return owner, repo


def _https_clone_url(repo_id: str) -> str:
    owner, repo = _split_repo_id(repo_id)
    return f"{GIN_HOST}/{owner}/{repo}.git"


def _raw_url(repo_id: str, rel: str, branch: str = "master") -> str:
    owner, repo = _split_repo_id(repo_id)
    rel_q = urllib.parse.quote(rel.lstrip("/"))
    return f"{GIN_HOST}/{owner}/{repo}/raw/{branch}/{rel_q}"


def _resolve_local_dir(
    repo_id: str,
    local_dir: Optional[str | Path] = None,
    spartan_detect: bool = True,
) -> Path:
    """Mirror the HuggingFace local-dir resolution conventions.

    Priority:
      1. explicit ``local_dir``
      2. Spartan project FS (``/data/gpfs/projects/punim*``)
      3. SciTeX runtime dir ``<scope-root>/runtime/gin/<owner>__<repo>/``
    """
    if local_dir:
        return Path(local_dir).expanduser().resolve()

    owner, repo = _split_repo_id(repo_id)
    safe = f"{owner}__{repo}"

    if spartan_detect:
        import glob as _glob
        spartan_projects = _glob.glob("/data/gpfs/projects/punim*") or \
                           _glob.glob("/data/projects/punim*")
        if spartan_projects:
            return Path(spartan_projects[0]) / "gin" / safe

    new_path = _runtime_dir() / "gin" / safe

    # Back-compat migration (matches huggingface.py §8 behaviour).
    old_path = _user_root() / "gin" / safe
    if old_path.exists() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)

    return new_path


def _parse_annex_pointer(path: Path) -> Optional[tuple[str, int]]:
    """Return ``(key, declared_size)`` if ``path`` is an annex pointer."""
    try:
        if path.stat().st_size > 4096:
            return None
        text = path.read_text(errors="replace").strip()
    except (OSError, UnicodeDecodeError):
        return None
    m = _ANNEX_POINTER_RE.match(text)
    if not m:
        return None
    return m.group("key"), int(m.group("size"))


def _git(cmd: list[str], cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(
        ["git", *cmd], cwd=cwd, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(cmd)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()}"
        )


def _http_get_with_size(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    timeout: float = 600.0,
) -> tuple[int, int]:
    """Stream ``GET url`` into ``dest``. Returns ``(content_length, bytes_written)``.

    Atomic via ``<dest>.partial`` + ``os.replace``. Idempotent: if
    ``dest`` already exists and is larger than 4 KiB (i.e. not an
    annex pointer), the GET is skipped.
    """
    if dest.exists() and dest.stat().st_size > 4096:
        if _parse_annex_pointer(dest) is None:
            return dest.stat().st_size, 0

    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)

    with _httpx.stream(
        "GET", url,
        timeout=timeout,
        headers={"User-Agent": "scitex-dataset/gin"},
        follow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        cl_hdr = resp.headers.get("Content-Length")
        cl = int(cl_hdr) if cl_hdr else -1
        bytes_written = 0
        with open(partial, "wb") as out:
            for chunk in resp.iter_bytes(chunk_size):
                if chunk:
                    out.write(chunk)
                    bytes_written += len(chunk)
    os.replace(partial, dest)
    return cl, bytes_written


# ----------------------------- public API ---------------------------- #

@supports_return_as
def gin_search(
    query: str = "",
    limit: int = 50,
    page: int = 1,
    logger=None,
) -> list[dict]:
    """Search GIN repos via ``/api/v1/repos/search``.

    Parameters
    ----------
    query : str
        Free-text query. Empty string returns the global feed.
    limit : int, default 50
        Server-side page size.
    page : int, default 1
        Page number, 1-indexed.
    logger : logging.Logger, optional
        Errors are logged here; otherwise silent (returns empty list).

    Returns
    -------
    list[dict]
        Raw repo records as returned by GIN. Run each through
        :func:`gin_format` for the package's normalized schema.
    """
    url = f"{GIN_API_V1}/repos/search"
    params = {"q": query, "limit": limit, "page": page}
    try:
        r = _httpx.get(url, params=params, timeout=30.0)
        r.raise_for_status()
    except _HTTPError as exc:
        if logger:
            logger.error(f"GIN search HTTP error: {exc}")
        return []
    payload = r.json()
    return payload.get("data", [])


@supports_return_as
def gin_info(repo_id: str, logger=None) -> dict:
    """Fetch metadata for one GIN repo via ``/api/v1/repos/{owner}/{repo}``.

    Returns the empty dict on HTTP error (logged if ``logger`` given).
    """
    owner, repo = _split_repo_id(repo_id)
    url = f"{GIN_API_V1}/repos/{owner}/{repo}"
    try:
        r = _httpx.get(url, timeout=30.0)
        r.raise_for_status()
    except _HTTPError as exc:
        if logger:
            logger.error(f"GIN info HTTP error for {repo_id}: {exc}")
        return {}
    return r.json()


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    batch_size: int = 50,
    logger=None,
) -> list[dict]:
    """Walk GIN's repo catalog by paginating ``/repos/search``.

    Mirrors the OpenNeuro / DANDI conventions so the unified search and
    database modules can index GIN uniformly.
    """
    out: list[dict] = []
    page = 1
    while True:
        batch = gin_search(query="", limit=batch_size, page=page, logger=logger)
        if not batch:
            break
        out.extend(batch)
        if logger:
            logger.info(f"GIN: fetched {len(out)} repos (page {page})")
        if max_datasets and len(out) >= max_datasets:
            break
        if len(batch) < batch_size:
            break
        page += 1
    return out[:max_datasets] if max_datasets else out


@supports_return_as
def gin_format(node: dict) -> dict:
    """Normalize a raw GIN repo record into the common dataset schema."""
    owner = (node.get("owner") or {}).get("login") or node.get("owner")
    name = node.get("name")
    full = node.get("full_name") or (f"{owner}/{name}" if owner and name else "")
    size_kb = node.get("size") or 0
    return {
        "id": full,
        "name": name,
        "owner": owner,
        "description": node.get("description"),
        "private": node.get("private"),
        "html_url": node.get("html_url"),
        "clone_url": node.get("clone_url"),
        "ssh_url": node.get("ssh_url"),
        "stars": node.get("stars_count"),
        "forks": node.get("forks_count"),
        "watchers": node.get("watchers_count"),
        "default_branch": node.get("default_branch"),
        "created": node.get("created_at"),
        "modified": node.get("updated_at"),
        "size_gb": round((size_kb * 1024) / (1024**3), 4) if size_kb else 0.0,
        "source": "gin",
    }


# Common alias matching the rest of the package.
format_dataset = gin_format


@supports_return_as
def gin_download(
    repo_id: str,
    local_dir: Optional[str | Path] = None,
    files: Optional[Iterable[str]] = None,
    branch: str = "master",
    prefer: str = "auto",
    metadata_only: bool = False,
    first_only: bool = False,
    logger=None,
) -> dict:
    """Download a GIN repo (metadata + annexed content).

    The default flow:

    1. ``git clone https://gin.g-node.org/<owner>/<repo>.git`` — pulls
       metadata and annex pointer text files (~MB; no auth).
    2. For each file matching ``files`` (default: every file under the
       repo whose worktree content is an annex pointer), stream the
       bytes from ``/raw/<branch>/<path>`` and replace the pointer
       atomically.

    Parameters
    ----------
    repo_id : str
        ``owner/repo`` GIN identifier (e.g.
        ``"USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM"``).
    local_dir : str | Path, optional
        Local destination. Falls back to the Spartan project FS or
        ``<scope-root>/runtime/gin/<owner>__<repo>/`` per
        :func:`_resolve_local_dir`.
    files : iterable of str, optional
        Repo-relative paths to fetch. ``None`` (default) fetches every
        annex pointer encountered.
    branch : str, default ``"master"``
        Source branch for the ``/raw/`` URL.
    prefer : {"auto", "https", "datalad"}, default ``"auto"``
        Backend selector. ``"datalad"`` requires the optional
        ``datalad`` extra; ``"auto"`` uses ``datalad`` if importable,
        else ``"https"``.  No silent fallback: an explicit
        ``"datalad"`` request raises if the optional dep is missing.
    metadata_only : bool, default ``False``
        Skip annex content; just clone the repo.
    first_only : bool, default ``False``
        Fetch only the FIRST pointer encountered (smoke-test path).
    logger : logging.Logger, optional
        Per-step progress is logged here if given.

    Returns
    -------
    dict
        ``{"repo_dir": Path, "files": [{"path": Path, "bytes": int}, ...],
           "total_bytes": int, "backend": str}``.

    Raises
    ------
    ImportError
        If ``prefer="datalad"`` is explicit and the optional dep is
        missing; the install hint can be read via
        :func:`scitex_dev.last_install_hint`.
    """
    owner, repo = _split_repo_id(repo_id)
    dest_root = _resolve_local_dir(repo_id, local_dir=local_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    repo_dir = dest_root / repo

    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)

    # ----- backend selection ------------------------------------------------
    backend = prefer
    if prefer == "auto":
        datalad = try_import_optional(
            "datalad.api",
            extra="datalad",
            pkg="scitex-dataset",
        )
        backend = "datalad" if datalad is not None else "https"
    elif prefer == "datalad":
        datalad = try_import_optional(
            "datalad.api",
            extra="datalad",
            pkg="scitex-dataset",
        )
        if datalad is None:
            raise ImportError(
                "prefer='datalad' requires the optional datalad extra. "
                "Install with: pip install 'scitex-dataset[datalad]'"
            )

    # ----- (1) metadata clone ----------------------------------------------
    if not (repo_dir / ".git").exists():
        _log(f"git clone {_https_clone_url(repo_id)} -> {repo_dir}")
        _git(["clone", _https_clone_url(repo_id), str(repo_dir)])
    else:
        _log(f"metadata already present: {repo_dir}")

    if metadata_only:
        return {
            "repo_dir": repo_dir,
            "files": [],
            "total_bytes": 0,
            "backend": "metadata",
        }

    # ----- (2) file selection ----------------------------------------------
    if files is not None:
        candidates = [repo_dir / Path(p) for p in files]
    else:
        candidates = [
            p for p in repo_dir.rglob("*")
            if p.is_file() and _parse_annex_pointer(p) is not None
        ]
        candidates.sort()
    if first_only:
        candidates = candidates[:1]

    # ----- (3) fetch content ------------------------------------------------
    if backend == "datalad":
        from datalad import api as _datalad_api  # type: ignore[import-not-found]
        rel_args = [str(p.relative_to(repo_dir).as_posix()) for p in candidates]
        _log(f"datalad get [{len(rel_args)} files] under {repo_dir}")
        _datalad_api.get(
            path=rel_args or None,
            dataset=str(repo_dir),
            jobs=4,
        )
    else:  # https — primary path
        for i, abs_p in enumerate(candidates, 1):
            rel = abs_p.relative_to(repo_dir)
            url = _raw_url(repo_id, rel.as_posix(), branch=branch)
            _log(f"GET [{i}/{len(candidates)}] {url}")
            cl, n = _http_get_with_size(url, abs_p)
            _log(f"  wrote {abs_p.stat().st_size:,} bytes "
                 f"(Content-Length={cl:,}; streamed={n:,})")

    sizes = [
        {"path": p, "bytes": p.stat().st_size if p.exists() else 0}
        for p in candidates
    ]
    total = sum(s["bytes"] for s in sizes)
    _log(f"GIN download complete: {len(sizes)} file(s), total {total:,} bytes")
    return {
        "repo_dir": repo_dir,
        "files": sizes,
        "total_bytes": total,
        "backend": backend,
    }


# EOF
