#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/bixbench.py

"""BixBench dataset preparation (cohort B).

Migrated from ``paper-scitex-clew/scripts/cohorts/b_bixbench/dataset/``
(operator brief 2026-06-12).

BixBench: 205 bioinformatics-analysis tasks across 54 capsules, hosted
on Hugging Face Hub as ``futurehouse/BixBench`` (public, no gate). The
HF snapshot bundles capsule code/data (``CapsuleData-<UUID>/`` +
``CapsuleNotebook-<UUID>/`` dirs) plus an answer-bearing
``BixBench.jsonl`` manifest.

Pipeline:

1. ``download(...)`` — ``huggingface_hub.snapshot_download(...)`` into
   ``capsule_dir``, then relocate ``BixBench.jsonl`` into the
   operator-private ``oracle_dir`` so the capsule volume stays
   answer-free (PR-ORACLE-B principle: nothing with oracle values lives
   under ``data/``).
2. ``mask(...)`` — read ``oracle_dir/BixBench.jsonl``, null the oracle
   fields (``answer``, ``ideal``, ``result``, ``distractors``,
   ``paper``), write ``benchmark_dir/questions.jsonl``. Preserves
   ``hypothesis`` (paper-level context, not an answer spoiler) and
   ``canary`` (leakage-detection sentinel — see the PR-MASK-REFINE
   note in the original script).

NOTE on compute: the HF snapshot is ~16 GB across 67 files; SLURM-only
on shared compute. ``mask(...)`` is pure-Python on the 205-record
JSONL (~285 KB) and safe to run anywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest

# Canonical cohort identity. ``COHORT_DIR`` matches the legacy on-disk
# layout used by the paper repo so a freshly-prepared dataset stays
# byte-compatible with anything that already points at
# ``data/cohort_b_bixbench``.
COHORT_ID = "bixbench"
COHORT_NAME = "BixBench"
COHORT_DIR = "cohort_b_bixbench"
HF_REPO_ID = "futurehouse/BixBench"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Upstream-with-answers manifest filename. Carries the oracle fields;
# moved into ``oracle_dir`` by ``download(...)``.
ORACLE_MANIFEST_NAME = "BixBench.jsonl"

# Oracle fields nulled in every record. ``hypothesis`` deliberately
# preserved — paper-level scientific context, not an answer spoiler
# (PR-MASK-REFINE in the original script). ``canary`` deliberately
# preserved — it's a deterministic sentinel for leakage detection;
# nulling it would defeat the mechanism, not improve isolation.
MASK_FIELDS = ("answer", "ideal", "result", "distractors", "paper")

# Backward-compat symlink retained so external scripts referencing the
# pre-rename ``BixBench_masked.jsonl`` keep working.
_COMPAT_SYMLINKS = (("BixBench_masked.jsonl", "questions.jsonl"),)


# ---------------------------------------------------------------------------
# Mask — pure-Python, network-free.
# ---------------------------------------------------------------------------


def mask_record(rec: dict) -> dict:
    """Return a new dict with oracle fields set to ``None``.

    Always sets the mask fields (even if absent in source) so the
    schema is uniform across all output records. Does not mutate the
    input. Idempotent: re-masking already-masked output yields
    identical bytes.
    """
    out = dict(rec)
    for k in MASK_FIELDS:
        out[k] = None
    return out


def _refresh_compat_symlinks(bench_dir: Path) -> None:
    for old_name, new_name in _COMPAT_SYMLINKS:
        link = bench_dir / old_name
        if link.is_symlink() or link.exists():
            link.unlink()
        # Relative target keeps the symlink host-portable.
        link.symlink_to(new_name)


def mask(
    *,
    oracle_dir: Path,
    benchmark_dir: Path,
    **_,
) -> dict:
    """Read the oracle manifest, write the agent-visible ``questions.jsonl``.

    Output is JSONL: one record per line, ``sort_keys=True`` and
    ``ensure_ascii=False`` for deterministic byte output across runs.
    Recreates the legacy ``BixBench_masked.jsonl`` backward-compat
    symlink.
    """
    src = oracle_dir / ORACLE_MANIFEST_NAME
    if not src.is_file():
        raise FileNotFoundError(
            f"bixbench: oracle manifest not found: {src}. "
            "Did you run download(...)?"
        )

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    dst = benchmark_dir / "questions.jsonl"
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            masked = mask_record(rec)
            fout.write(json.dumps(masked, sort_keys=True, ensure_ascii=False))
            fout.write("\n")
            n += 1

    _refresh_compat_symlinks(benchmark_dir)
    return {
        "output": str(dst),
        "n_records": n,
        "mask_fields": list(MASK_FIELDS),
        "compat_symlinks": [
            str(benchmark_dir / old) for old, _ in _COMPAT_SYMLINKS
        ],
    }


# ---------------------------------------------------------------------------
# Download — HF snapshot + oracle relocation.
# ---------------------------------------------------------------------------


def _relocate_oracle_manifest(local: Path, oracle: Path) -> str:
    """Move ``local`` → ``oracle`` idempotently.

    Returns a short status string for the caller's summary dict.
    Handles all four states (local-only / oracle-only / both / neither).
    """
    oracle.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(oracle.parent, 0o700)
    except OSError:  # pragma: no cover — perm denied on shared FS
        pass

    if local.is_file() and oracle.is_file():
        # Both copies present — keep oracle authoritative, drop the
        # /work duplicate so re-runs converge to a single copy. Only
        # remove the local copy if its bytes match the oracle.
        if local.read_bytes() == oracle.read_bytes():
            local.unlink()
            return "removed-duplicate-local-copy"
        raise RuntimeError(
            f"bixbench: oracle manifest mismatch between {local} and "
            f"{oracle}. Inspect both, pick one, delete the other."
        )
    if local.is_file() and not oracle.is_file():
        local.replace(oracle)
        return "moved-local-to-oracle"
    if oracle.is_file():
        return "already-relocated"
    raise FileNotFoundError(
        f"bixbench: {ORACLE_MANIFEST_NAME} missing from both "
        f"{local} and {oracle} after HF snapshot."
    )


def download(
    *,
    oracle_dir: Path,
    capsule_dir: Path,
    hf_token: str | None = None,
    max_workers: int = 4,
    **_,
) -> dict:
    """Pull the BixBench HF snapshot, relocate the oracle manifest.

    HEAVY: ~16 GB across ~67 files; SLURM-only on shared compute.
    ``huggingface_hub`` is an optional extra (``scitex-dataset[mcp]``
    doesn't include it) — install ``scitex-dataset[huggingface]`` or
    ``scitex-dataset[all]`` first.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover — exercised via tests
        raise RuntimeError(
            "bixbench.download requires huggingface_hub. "
            "Install with: pip install 'scitex-dataset[huggingface]'"
        ) from exc

    capsule_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(capsule_dir),
        max_workers=max_workers,
        token=hf_token,
    )

    local_manifest = capsule_dir / ORACLE_MANIFEST_NAME
    oracle_manifest = oracle_dir / ORACLE_MANIFEST_NAME
    status = _relocate_oracle_manifest(local_manifest, oracle_manifest)
    return {
        "capsule_dir": str(capsule_dir),
        "oracle_manifest": str(oracle_manifest),
        "relocation_status": status,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def prepare(
    *,
    paths: BenchmarkPaths | None = None,
    oracle_root: Path | str | None = None,
    dataset_root: Path | str | None = None,
    version: str = "v0-unstamped",
    skip_download: bool = False,
    **_,
) -> dict:
    """Run the full BixBench preparation pipeline.

    Set ``skip_download=True`` to skip the HF snapshot pull (useful if
    the oracle manifest has already been hand-staged).
    """
    if paths is None:
        paths = resolve_paths(
            COHORT_DIR, oracle_root=oracle_root, dataset_root=dataset_root
        )

    out: dict = {"cohort": COHORT_ID, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            oracle_dir=paths.oracle_dir, capsule_dir=paths.capsule_dir
        )
    out["mask"] = mask(
        oracle_dir=paths.oracle_dir, benchmark_dir=paths.benchmark_dir
    )

    manifest_path = write_manifest(
        manifest_dir=paths.manifest_dir,
        id=COHORT_ID,
        name=COHORT_NAME,
        version=version,
        source_url=SOURCE_URL,
        cohort_dir=COHORT_DIR,
        tracked_paths=[Path(out["mask"]["output"])],
        tracked_root=paths.benchmark_dir,
        mask_seed="",
    )
    out["manifest"] = str(manifest_path)
    return out


__all__ = [
    "COHORT_ID",
    "COHORT_NAME",
    "COHORT_DIR",
    "HF_REPO_ID",
    "HF_REPO_TYPE",
    "SOURCE_URL",
    "MASK_FIELDS",
    "ORACLE_MANIFEST_NAME",
    "download",
    "mask",
    "mask_record",
    "prepare",
]

# EOF
