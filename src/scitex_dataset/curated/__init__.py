#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/curated/__init__.py

"""CURATED AI-for-science datasets: where they came from, how, and what is hidden.

A curated dataset is an AI-for-science benchmark prepared by
:mod:`scitex_dataset.ai_for_science` (``download -> standardize``) into three
roles under ``<dataset-root>/ai-for-science/<benchmark>/``:

``raw/``
    the upstream public snapshot as-is. It CONTAINS the answers.
``for_solver/``
    the agent-visible, answer-masked view: tasks without answers, run outputs
    and logs stripped, answer values redacted, hosting identifiers and
    source-revealing path names neutralised.
``eval/``
    the operator-side scorer view: the held-out answers and the scorer.

and a provenance manifest ``.scitex/dataset/MANIFEST.yaml`` (source URL,
snapshot version, mask seed, the SHA-256 of every tracked agent-visible file,
the scitex-dataset version that prepared it). That manifest is what a Clew
verification pins: a drifted input no longer matches its recorded hash.

:func:`describe` turns one benchmark tree into what a page shows: the source,
the derivation chain (source -> masking step with its policy versions ->
artifacts with checksums) and a MASKING SUMMARY. The summary says WHAT is
masked or held out — categories, counts, sizes — and never the content.
``raw/`` and ``eval/`` file names are listed only for ``operator=True``
(the host decides who that is); their bytes are never read for display and
nothing here returns a file's content.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._sources import AGENTIC_BENCH_SOURCES, SOURCE_INFO
from ..catalog import human_size

__all__ = ["CURATED_BENCHMARKS", "describe", "list_curated"]

CURATED_BENCHMARKS = tuple(AGENTIC_BENCH_SOURCES)

#: Files shown per role at most.
_MAX_LISTED = 200
_MAX_WALKED = 100_000


def _dataset_root(dataset_root=None) -> Path:
    from ..ai_for_science._base import _dataset_root as resolve

    return Path(resolve(dataset_root))


def _paths(benchmark: str, dataset_root=None):
    from ..ai_for_science._base import resolve_paths

    if benchmark not in CURATED_BENCHMARKS:
        raise KeyError(benchmark)
    return resolve_paths(benchmark, dataset_root=_dataset_root(dataset_root))


def _inventory(top: Path, *, names: bool) -> Dict[str, Any]:
    """File count and bytes under ``top``; names only when asked. Never opens a file."""
    count = 0
    size = 0
    listed: List[Dict[str, Any]] = []
    if top.is_dir():
        for current, dirs, files in os.walk(top, followlinks=False):
            dirs.sort()
            for name in sorted(files):
                path = Path(current) / name
                try:
                    stat = path.lstat()
                except OSError:
                    continue
                count += 1
                size += stat.st_size
                if names and len(listed) < _MAX_LISTED:
                    listed.append(
                        {
                            "path": path.relative_to(top).as_posix(),
                            "size_bytes": stat.st_size,
                            "size": human_size(stat.st_size),
                        }
                    )
                if count >= _MAX_WALKED:
                    break
    return {
        "present": top.is_dir(),
        "files": count,
        "size_bytes": size,
        "size": human_size(size),
        "listing": listed if names else None,
        "truncated": names and count > len(listed),
    }


def _masking_policies() -> List[Dict[str, str]]:
    from ..ai_for_science._identifiers import IDENTIFIER_POLICY
    from ..ai_for_science._leak_strip import LEAK_DIRS
    from ..ai_for_science._materialization_identity import _MATERIALIZATION_CONTRACT
    from ..ai_for_science._neutral_paths import PATH_POLICY
    from ..ai_for_science._value_match import VALUE_MATCH_POLICY

    return [
        {
            "name": "Answers held out",
            "policy": "for_solver/eval split",
            "what": "Ground-truth answers and rubrics move to eval/ and never reach the agent view.",
        },
        {
            "name": "Run outputs and logs stripped",
            "policy": "leak-dirs: " + ", ".join(LEAK_DIRS),
            "what": "The authors' original outputs and running logs are removed at any depth.",
        },
        {
            "name": "Answer values redacted",
            "policy": VALUE_MATCH_POLICY,
            "what": "Answer values found in the remaining text are redacted, precision-aware; a residual match fails the preparation.",
        },
        {
            "name": "Hosting identifiers redacted",
            "policy": IDENTIFIER_POLICY,
            "what": "Identifiers that would reveal the source record are removed.",
        },
        {
            "name": "Neutral path names",
            "policy": PATH_POLICY,
            "what": "Path names that carry source identifiers are neutralised.",
        },
        {
            "name": "Cache identity",
            "policy": _MATERIALIZATION_CONTRACT,
            "what": "A prepared copy is reused only while every policy above is unchanged.",
        },
    ]


def list_curated(dataset_root=None) -> List[Dict[str, Any]]:
    """The curated benchmarks prepared on this host (those with a manifest)."""
    out = []
    for benchmark in CURATED_BENCHMARKS:
        paths = _paths(benchmark, dataset_root)
        manifest_path = paths.manifest_dir / "MANIFEST.yaml"
        if not manifest_path.is_file():
            continue
        from ..ai_for_science._manifest import read_manifest

        manifest = read_manifest(manifest_path)
        out.append(
            {
                "benchmark": benchmark,
                "title": manifest.get("name") or SOURCE_INFO[benchmark]["name"],
                "version": manifest.get("version"),
                "source_url": manifest.get("source_url"),
                "prepared_at": manifest.get("prepared_at"),
                "description": SOURCE_INFO[benchmark]["description"],
            }
        )
    return out


def describe(
    benchmark: str, dataset_root=None, *, operator: bool = False
) -> Optional[Dict[str, Any]]:
    """What one curated benchmark page shows; ``None`` if it is not prepared.

    ``operator=False`` (everyone): source, chain, policies, agent-visible
    files with SHA-256, and COUNTS of the hidden roles. ``operator=True``
    adds the names and sizes of the hidden files. No mode returns content.
    """
    from ..ai_for_science._manifest import read_manifest

    paths = _paths(benchmark, dataset_root)
    manifest_path = paths.manifest_dir / "MANIFEST.yaml"
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    info = SOURCE_INFO[benchmark]
    tracked = [
        {**f, "size": human_size(f.get("size"))} for f in manifest.get("files", [])
    ]
    raw = _inventory(paths.raw_dir, names=operator)
    held_out = _inventory(paths.eval_dir, names=operator)
    agent_view = _inventory(paths.for_solver_dir, names=False)
    source = {
        "name": info["name"],
        "homepage": manifest.get("source_url") or info["url"],
        "version": manifest.get("version"),
        "format": info.get("format"),
    }
    chain = [
        {
            "kind": "source",
            "label": f"{source['name']} public snapshot",
            "detail": f"version {source['version']}",
            "homepage": source["homepage"],
            "role": "raw/",
            "visibility": "operator only",
            "files": raw["files"],
            "size": raw["size"],
        },
        {
            "kind": "step",
            "label": "standardize (mask and split)",
            "detail": (
                f"scitex-dataset {manifest.get('scitex_dataset_version') or '(dev)'}, "
                f"mask seed {manifest.get('mask_seed') or '(none: deterministic)'}, "
                f"prepared {manifest.get('prepared_at')}"
            ),
        },
        {
            "kind": "artifact",
            "label": "Agent-visible masked view",
            "role": "for_solver/",
            "visibility": "agents and everyone",
            "files": agent_view["files"],
            "size": agent_view["size"],
            "tracked": len(tracked),
        },
        {
            "kind": "artifact",
            "label": "Held-out answers and scorer",
            "role": "eval/",
            "visibility": "operator only",
            "files": held_out["files"],
            "size": held_out["size"],
        },
    ]
    return {
        "benchmark": benchmark,
        "title": manifest.get("name") or info["name"],
        "description": info["description"],
        "source": source,
        "manifest": {
            "path": ".scitex/dataset/MANIFEST.yaml",
            "id": manifest.get("id"),
            "version": manifest.get("version"),
            "mask_seed": manifest.get("mask_seed"),
            "prepared_at": manifest.get("prepared_at"),
            "scitex_dataset_version": manifest.get("scitex_dataset_version"),
        },
        "chain": chain,
        "policies": _masking_policies(),
        "agent_visible": {"tracked": tracked[:_MAX_LISTED], **agent_view},
        "held_out": held_out,
        "raw": raw,
        "operator": operator,
    }


# EOF
