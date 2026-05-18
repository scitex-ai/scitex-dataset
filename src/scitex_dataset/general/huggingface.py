#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-05 12:00:00 (ywatanabe)"
# File: src/scitex_dataset/general/huggingface.py

"""
HuggingFace Hub client for dataset and model downloads.

HuggingFace Hub (https://huggingface.co) hosts large language models, vision models,
and datasets. This module provides utilities to fetch, search, and download datasets
from HuggingFace with project-FS awareness (on Spartan, caches to /data/gpfs/ instead
of home to avoid quota issues).

API Documentation: https://huggingface.co/docs/hub/
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Note: supports_return_as is a decorator from scitex_dev that formats output
# It's optional and mainly for consistency with other modules
try:
    from scitex_dev import supports_return_as
except ImportError:
    # Fallback: define a no-op decorator
    def supports_return_as(fn):
        return fn


__all__ = [
    "fetch_dataset",
    "search_datasets",
    "search_hub",
    "fetch_all_datasets",
    "format_dataset",
    "dataset_info",
    "download_file",
]


def _resolve_token(gated_token_var: str = "HF_TOKEN_PATH") -> Optional[str]:
    """
    Resolve HuggingFace token from environment chain.

    Token resolution priority:
    1. HF_TOKEN environment variable
    2. Path in HF_TOKEN_PATH (or gated_token_var)
    3. ~/.bash.d/secrets/access_tokens/huggingface.txt
    4. None (use unauthenticated access)

    Parameters
    ----------
    gated_token_var : str
        Environment variable name pointing to token file (default: HF_TOKEN_PATH).

    Returns
    -------
    str or None
        The resolved token, or None if no token found.
    """
    # Priority 1: HF_TOKEN env var (direct token)
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]

    # Priority 2: HF_TOKEN_PATH env var (path to token file)
    token_path_env = os.environ.get(gated_token_var)
    if token_path_env and Path(token_path_env).exists():
        try:
            return Path(token_path_env).read_text().strip()
        except Exception:
            pass

    # Priority 3: Default secret location
    default_secret = (
        Path.home() / ".bash.d" / "secrets" / "access_tokens" / "huggingface.txt"
    )
    if default_secret.exists():
        try:
            return default_secret.read_text().strip()
        except Exception:
            pass

    return None


def _resolve_local_dir(
    repo_id: str,
    local_dir: Optional[str] = None,
    spartan_detect: bool = True,
) -> Path:
    """
    Resolve local directory for dataset download.

    Priority:
    1. Explicit local_dir parameter
    2. Spartan project filesystem: /data/gpfs/projects/<punim>/<repo_id>/
    3. Home directory: ~/.scitex/dataset/huggingface/<repo_id>/

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g., "username/dataset_name").
    local_dir : str, optional
        Explicit local directory.
    spartan_detect : bool
        Auto-detect Spartan filesystem (default: True).

    Returns
    -------
    Path
        Resolved local directory path.
    """
    if local_dir:
        return Path(local_dir).expanduser().resolve()

    # Try to detect Spartan project filesystem
    if spartan_detect:
        try:
            import glob as glob_module

            spartan_projects = glob_module.glob("/data/gpfs/projects/punim*")
            if spartan_projects:
                # Use first matching project directory
                project_dir = Path(spartan_projects[0])
                repo_name = repo_id.replace("/", "_")
                return project_dir / repo_name

                # Note: for actual BMB use, caller should specify explicit local_dir
        except Exception:
            pass

    # Fall back to home directory
    repo_name = repo_id.replace("/", "_")
    return Path.home() / ".scitex" / "dataset" / "huggingface" / repo_name


@supports_return_as
def fetch_dataset(
    repo_id: str,
    local_dir: Optional[str] = None,
    repo_type: str = "dataset",
    gated_token_var: str = "HF_TOKEN_PATH",
    max_workers: int = 4,
    hf_home_override: Optional[str] = None,
) -> Path:
    """
    Fetch a complete HuggingFace dataset to disk.

    On Spartan, if hf_home_override is provided, sets HF_HOME to that directory
    so the content-addressed cache doesn't grow home quotas.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g., "Anthropic/BioMysteryBench-full").
    local_dir : str, optional
        Local directory for dataset. If None, uses Spartan project FS if detected,
        else ~/.scitex/dataset/huggingface/<repo_id>/.
    repo_type : str
        Repository type: "dataset" (default) or "model".
    gated_token_var : str
        Environment variable name for token file path (default: HF_TOKEN_PATH).
    max_workers : int
        Parallel download workers (default: 4).
    hf_home_override : str, optional
        Override HF_HOME cache directory (recommended on Spartan).

    Returns
    -------
    Path
        Path to the downloaded dataset directory.

    Raises
    ------
    Exception
        If token resolution fails for gated repositories or network errors occur.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install: pip install huggingface-hub"
        )

    # Resolve token if needed
    token = _resolve_token(gated_token_var=gated_token_var)

    # Resolve local directory
    local_dir_path = _resolve_local_dir(repo_id, local_dir=local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME if override provided
    original_hf_home = None
    if hf_home_override:
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(Path(hf_home_override).expanduser().resolve())

    try:
        # Perform snapshot download
        result_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=str(local_dir_path),
            token=token,
            max_workers=max_workers,
            force_download=False,
            resume_download=True,
        )
        return Path(result_path)
    finally:
        # Restore original HF_HOME
        if hf_home_override and original_hf_home is not None:
            os.environ["HF_HOME"] = original_hf_home
        elif hf_home_override and "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]


@supports_return_as
def search_datasets(
    query: str,
    limit: int = 50,
) -> List[Dict]:
    """
    Search for datasets on HuggingFace.

    Parameters
    ----------
    query : str
        Search query string.
    limit : int
        Maximum number of results (default: 50).

    Returns
    -------
    list[dict]
        List of search result dictionaries with fields:
        id, name, description, likes, downloads, private, gated, etc.
    """
    try:
        from huggingface_hub import list_datasets
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install: pip install huggingface-hub"
        )

    results = []
    for dataset_info in list_datasets(search=query, limit=limit, full=False):
        results.append(
            {
                "id": dataset_info.id,
                "name": dataset_info.id.split("/")[-1],
                "description": dataset_info.description or "",
                "downloads": dataset_info.downloads or 0,
                "likes": dataset_info.likes or 0,
                "private": dataset_info.private or False,
                "gated": dataset_info.gated or False,
                "url": f"https://huggingface.co/datasets/{dataset_info.id}",
            }
        )

    return results


@supports_return_as
def dataset_info(
    repo_id: str,
    repo_type: str = "dataset",
) -> Dict:
    """
    Get metadata about a HuggingFace dataset or model.

    Parameters
    ----------
    repo_id : str
        Repository ID (e.g., "username/dataset_name").
    repo_type : str
        Repository type: "dataset" (default) or "model".

    Returns
    -------
    dict
        Dataset metadata: id, name, description, downloads, likes, private,
        gated, size_gb, created_at, last_modified, etc.
    """
    try:
        from huggingface_hub import dataset_info as hf_dataset_info
        from huggingface_hub import model_info as hf_model_info
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install: pip install huggingface-hub"
        )

    try:
        if repo_type == "dataset":
            info = hf_dataset_info(repo_id=repo_id)
        elif repo_type == "model":
            info = hf_model_info(repo_id=repo_id)
        else:
            raise ValueError(f"Unknown repo_type: {repo_type}")

        # Extract key fields
        return {
            "id": info.id,
            "name": info.id.split("/")[-1],
            "description": info.description or "",
            "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
            "private": info.private or False,
            "gated": getattr(info, "gated", False) or False,
            "url": f"https://huggingface.co/{repo_type}s/{repo_id}",
            "created_at": str(info.created_at) if hasattr(info, "created_at") else None,
            "last_modified": str(info.last_modified)
            if hasattr(info, "last_modified")
            else None,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch info for {repo_id}: {e}") from e


@supports_return_as
def download_file(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
    repo_type: str = "dataset",
) -> Path:
    """
    Download a single file from a HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        Repository ID (e.g., "username/dataset_name").
    filename : str
        Path within the repository (e.g., "data/train.csv").
    local_dir : str, optional
        Local directory for download. If None, uses ~/.scitex/dataset/huggingface/<repo_id>/.
    repo_type : str
        Repository type: "dataset" (default) or "model".

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    Exception
        If download fails.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install: pip install huggingface-hub"
        )

    # Resolve local directory
    local_dir_path = _resolve_local_dir(repo_id, local_dir=local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    token = _resolve_token()

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=str(local_dir_path),
            local_dir=None,  # Use cache_dir
            token=token,
            force_download=False,
            resume_download=True,
        )
        return Path(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {e}") from e


# Clearer alias — peer modules also expose ``search_datasets``, but at
# the package level that name belongs to ``scitex_dataset.search`` (the
# in-memory filter). New code should prefer ``search_hub``.
search_hub = search_datasets


def format_dataset(ds: Dict) -> Dict:
    """Normalize an HF search-result dict to the common dataset schema.

    HuggingFace records lack the n_subjects / modalities / tasks fields
    that BIDS/NWB sources expose, so those keys are emitted as ``None``
    or empty lists. ``downloads`` and ``likes`` are preserved.
    """
    return {
        "id": ds.get("id"),
        "name": ds.get("name") or (ds.get("id") or "").split("/")[-1],
        "description": ds.get("description", ""),
        "readme": ds.get("description", ""),
        "downloads": ds.get("downloads", 0) or 0,
        "likes": ds.get("likes", 0) or 0,
        "n_subjects": None,
        "modalities": [],
        "tasks": [],
        "size_gb": None,
        "private": ds.get("private", False) or False,
        "gated": ds.get("gated", False) or False,
        "url": ds.get("url"),
        "source": "huggingface",
    }


def fetch_all_datasets(
    query: str = "",
    max_datasets: Optional[int] = None,
    logger=None,
    **_unused,
) -> List[Dict]:
    """Catalog-style adapter so HuggingFace can plug into ``database.build``.

    Unlike OpenNeuro/DANDI/etc., HuggingFace has no bounded catalog —
    ``query`` is required for meaningful results. Without one this calls
    ``search_hub("")`` which lists by recency up to ``max_datasets``.

    Parameters
    ----------
    query : str
        Search query. Empty string lists by recency (HF default).
    max_datasets : int, optional
        Cap on results. Default 1000 to avoid runaway indexing.
    """
    limit = max_datasets if max_datasets and max_datasets > 0 else 1000
    if logger:
        logger.info(f"Searching HuggingFace Hub (query={query!r}, limit={limit})...")
    return search_hub(query=query, limit=limit)


# EOF
