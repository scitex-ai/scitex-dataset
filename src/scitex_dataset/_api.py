#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_api.py

"""Per-source ``<src>_fetch`` / ``<src>_format`` aliases for MCP parity.

Each MCP tool ``dataset_<src>_fetch`` has a matching Python callable
``<src>_fetch`` exported from the package root, so the
``scitex-dev ecosystem audit-mcp-tools`` § 6 parity check passes.

The aliases are thin re-bindings of the canonical
``<domain>.<src>.fetch_all_datasets`` / ``format_dataset`` functions —
no logic of their own.
"""

from __future__ import annotations

from .biology.geo import fetch_all_datasets as geo_fetch
from .biology.geo import format_dataset as geo_format
from .general.figshare import fetch_all_datasets as figshare_fetch
from .general.figshare import format_dataset as figshare_format
from .general.huggingface import dataset_info as huggingface_info
from .general.huggingface import download_file as huggingface_download_file
from .general.huggingface import fetch_all_datasets as huggingface_fetch
from .general.huggingface import format_dataset as hf_format
from .general.huggingface import search_hub as huggingface_search
from .general.openml import fetch_all_datasets as openml_fetch
from .general.openml import format_dataset as openml_format
from .general.zenodo import fetch_all_datasets as zenodo_fetch
from .general.zenodo import format_dataset as zenodo_format
from .medical.clinicaltrials import fetch_all_datasets as clinicaltrials_fetch
from .medical.clinicaltrials import format_dataset as clinicaltrials_format
from .neuroscience.dandi import fetch_all_datasets as dandi_fetch
from .neuroscience.dandi import format_dataset as dandi_format
from .neuroscience.gin import fetch_all_datasets as gin_fetch
from .neuroscience.gin import format_dataset as gin_format
from .neuroscience.gin import gin_download
from .neuroscience.gin import gin_info
from .neuroscience.gin import gin_search
from .neuroscience.openneuro import fetch_all_datasets as openneuro_fetch
from .neuroscience.openneuro import format_dataset as openneuro_format
from .neuroscience.physionet import fetch_all_datasets as physionet_fetch
from .neuroscience.physionet import format_dataset as physionet_format
from .pharmacology.chembl import fetch_all_datasets as chembl_fetch
from .pharmacology.chembl import format_dataset as chembl_format
from .pharmacology.moleculenet import fetch_all_datasets as moleculenet_fetch
from .pharmacology.moleculenet import format_dataset as moleculenet_format

__all__ = [
    "openneuro_fetch",
    "openneuro_format",
    "dandi_fetch",
    "dandi_format",
    "physionet_fetch",
    "physionet_format",
    "gin_fetch",
    "gin_format",
    "gin_search",
    "gin_info",
    "gin_download",
    "download_dataset",
    "zenodo_fetch",
    "zenodo_format",
    "figshare_fetch",
    "figshare_format",
    "openml_fetch",
    "openml_format",
    "moleculenet_fetch",
    "moleculenet_format",
    "geo_fetch",
    "geo_format",
    "chembl_fetch",
    "chembl_format",
    "clinicaltrials_fetch",
    "clinicaltrials_format",
    "huggingface_fetch",
    "hf_format",
    "huggingface_search",
    "huggingface_info",
    "huggingface_download_file",
]


# ---------------------------------------------------------------------- #
# Unified download dispatcher (issue #36).
#
# A single ``download_dataset(source, id, dest, **opts)`` entry point
# that routes to the per-source downloader. Today only the sources
# that ship a real download path (HuggingFace, GIN) are wired up;
# catalog-only sources raise ``NotImplementedError`` with a hint.
# ---------------------------------------------------------------------- #

def download_dataset(
    source: str,
    id: str,
    dest=None,
    **kwargs,
):
    """Unified ``download_dataset(source, id, dest, **opts)`` dispatcher.

    Parameters
    ----------
    source : str
        Source id, case-insensitive. Matched against
        :data:`scitex_dataset._sources.ALL_SOURCES`.
    id : str
        Source-native identifier — HF ``org/name``, GIN ``owner/repo``,
        etc.
    dest : str | Path, optional
        Local destination. Per-source default applies if ``None``.
    **kwargs
        Forwarded to the underlying ``<source>_download`` function.

    Returns
    -------
    Any
        Whatever the underlying downloader returns (path or manifest
        dict).

    Raises
    ------
    ValueError
        If ``source`` is not in the registry.
    NotImplementedError
        If the matched source has no download adapter wired up yet.
    """
    src = source.lower().strip()
    if src in ("gin",):
        return gin_download(repo_id=id, local_dir=dest, **kwargs)
    if src in ("huggingface", "hf"):
        return huggingface_fetch(repo_id=id, local_dir=dest, **kwargs)

    from ._sources import ALL_SOURCES
    if src in ALL_SOURCES:
        raise NotImplementedError(
            f"download_dataset: source {src!r} is catalog-only today; "
            f"a download adapter has not yet been wired up. "
            f"Track at https://github.com/ywatanabe1989/scitex-dataset/issues"
        )
    raise ValueError(
        f"download_dataset: unknown source {source!r}. "
        f"Known sources: {sorted(ALL_SOURCES)}"
    )


# EOF
