#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/biomysterybench.py

"""BioMysteryBench dataset preparation (cohort C).

Migrated from
``paper-scitex-clew/scripts/cohorts/c_biomysterybench/dataset/``
(operator brief 2026-06-12).

BioMysteryBench: biology-mystery problems with rubric-graded answers,
hosted on Hugging Face Hub by Anthropic. Two variants:

- ``Anthropic/BioMysteryBench-preview``  — 5 problems, ~11 MB, public.
- ``Anthropic/BioMysteryBench-full``     — 99 problems, ~159 GB, gated.

Pipeline:

1. ``download(...)`` — snapshot-download the preview tree (default) or
   the full set when ``download_full=True`` AND HF access is granted.
   Relocates upstream-with-answers artifacts (``problems.csv``,
   ``problems.parquet``, ``README.md``) into the operator-private
   ``oracle_dir``.
2. ``mask(...)`` — read ``oracle_dir/problems.csv``, null the
   ``answer_rubric`` column, write JSONL to
   ``benchmark_dir/questions.jsonl``. All other columns preserved
   verbatim.

NOTE on compute: the preview is small (~11 MB), the full set is multi-
GB and must run on SLURM. ``mask(...)`` is pure-Python on a small CSV
and safe to run anywhere.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest

# Canonical cohort identity. ``COHORT_DIR`` matches the legacy on-disk
# layout used by the paper repo.
COHORT_ID = "biomysterybench"
COHORT_NAME = "BioMysteryBench"
COHORT_DIR = "cohort_c_biomysterybench"

HF_REPO_ID_PREVIEW = "Anthropic/BioMysteryBench-preview"
HF_REPO_ID_FULL = "Anthropic/BioMysteryBench-full"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID_PREVIEW}"

# Columns kept verbatim from problems.csv.
PRESERVED_FIELDS = ("id", "question", "allowed_domains", "human_solvable")
# Column whose value is replaced with None.
ORACLE_FIELD = "answer_rubric"

# Upstream-with-answers artifacts relocated to oracle_dir after a
# successful HF snapshot.
ORACLE_FILENAMES = ("problems.csv", "problems.parquet", "README.md")

# Backward-compat symlink for the pre-rename name.
_COMPAT_SYMLINKS = (("problems_masked.jsonl", "questions.jsonl"),)


# ---------------------------------------------------------------------------
# Mask — pure-Python, network-free.
# ---------------------------------------------------------------------------


def mask_row(row: dict) -> dict:
    """Project ``row`` to ``{*PRESERVED_FIELDS: ..., ORACLE_FIELD: None}``.

    Anything in ``row`` that isn't a preserved field is dropped — the
    mask is deliberately strict so an upstream schema drift can't
    accidentally leak a new answer-bearing column. Idempotent
    (re-masking already-masked rows yields identical bytes).
    """
    out = {field: row.get(field) for field in PRESERVED_FIELDS}
    out[ORACLE_FIELD] = None
    return out


def _refresh_compat_symlinks(bench_dir: Path) -> None:
    for old_name, new_name in _COMPAT_SYMLINKS:
        link = bench_dir / old_name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(new_name)


def mask(
    *,
    oracle_dir: Path,
    benchmark_dir: Path,
    **_,
) -> dict:
    """Read ``oracle_dir/problems.csv``, write masked ``questions.jsonl``.

    Output is JSONL (one JSON object per line, ``sort_keys=True``,
    ``ensure_ascii=False``) — symmetric with cohort B and deliberately
    distinct from the upstream CSV so the on-disk shape doubles as a
    hint that the file has been masked.
    """
    src = oracle_dir / "problems.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"biomysterybench: oracle source not found: {src}. "
            "Did you run download(...)?"
        )

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    dst = benchmark_dir / "questions.jsonl"
    n = 0
    with src.open("r", encoding="utf-8", newline="") as fh_in, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as fh_out:
        reader = csv.DictReader(fh_in)
        for row in reader:
            fh_out.write(
                json.dumps(mask_row(row), sort_keys=True, ensure_ascii=False)
            )
            fh_out.write("\n")
            n += 1

    _refresh_compat_symlinks(benchmark_dir)
    return {
        "output": str(dst),
        "n_records": n,
        "mask_fields": [ORACLE_FIELD],
        "compat_symlinks": [
            str(benchmark_dir / old) for old, _ in _COMPAT_SYMLINKS
        ],
    }


# ---------------------------------------------------------------------------
# Download — HF snapshot + oracle relocation.
# ---------------------------------------------------------------------------


def _relocate_oracle_files(
    capsule_dir: Path, oracle_dir: Path
) -> list[str]:
    """Move each oracle-bearing file from capsule_dir to oracle_dir.

    Returns the list of relocation statuses (one per file). Idempotent.
    """
    oracle_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[str] = []
    for name in ORACLE_FILENAMES:
        local = capsule_dir / name
        oracle = oracle_dir / name
        if local.is_file() and oracle.is_file():
            if local.read_bytes() == oracle.read_bytes():
                local.unlink()
                statuses.append(f"{name}: removed-duplicate-local-copy")
            else:
                raise RuntimeError(
                    f"biomysterybench: oracle file mismatch between {local} "
                    f"and {oracle}. Inspect both, pick one, delete the other."
                )
        elif local.is_file() and not oracle.is_file():
            local.replace(oracle)
            statuses.append(f"{name}: moved-local-to-oracle")
        elif oracle.is_file():
            statuses.append(f"{name}: already-relocated")
        else:
            # Not every upstream variant ships all three (preview has
            # CSV + README only). Record absent files so callers know
            # which path was taken.
            statuses.append(f"{name}: absent-upstream")
    return statuses


def download(
    *,
    oracle_dir: Path,
    capsule_dir: Path,
    download_full: bool = False,
    hf_token: str | None = None,
    max_workers: int = 4,
    **_,
) -> dict:
    """Pull the BMB HF snapshot, relocate the oracle artifacts.

    Preview snapshot (~11 MB, public) is always attempted. When
    ``download_full=True`` and HF access is granted the full set
    (~159 GB, gated) is pulled afterwards. ``huggingface_hub`` is
    required.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "biomysterybench.download requires huggingface_hub. "
            "Install with: pip install 'scitex-dataset[huggingface]'"
        ) from exc

    capsule_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []

    snapshot_download(
        repo_id=HF_REPO_ID_PREVIEW,
        repo_type=HF_REPO_TYPE,
        local_dir=str(capsule_dir),
        max_workers=max_workers,
        token=hf_token,
    )
    pulled.append(HF_REPO_ID_PREVIEW)

    if download_full:
        snapshot_download(
            repo_id=HF_REPO_ID_FULL,
            repo_type=HF_REPO_TYPE,
            local_dir=str(capsule_dir),
            max_workers=max_workers,
            token=hf_token,
        )
        pulled.append(HF_REPO_ID_FULL)

    statuses = _relocate_oracle_files(capsule_dir, oracle_dir)
    return {
        "capsule_dir": str(capsule_dir),
        "oracle_dir": str(oracle_dir),
        "snapshots_pulled": pulled,
        "relocations": statuses,
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
    download_full: bool = False,
    **_,
) -> dict:
    """Run the full BioMysteryBench preparation pipeline.

    Pass ``download_full=True`` to attempt the gated 159 GB set after
    the preview. Pass ``skip_download=True`` to skip both HF pulls if
    the oracle CSV has already been hand-staged.
    """
    if paths is None:
        paths = resolve_paths(
            COHORT_DIR, oracle_root=oracle_root, dataset_root=dataset_root
        )

    out: dict = {"cohort": COHORT_ID, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            oracle_dir=paths.oracle_dir,
            capsule_dir=paths.capsule_dir,
            download_full=download_full,
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
    "HF_REPO_ID_PREVIEW",
    "HF_REPO_ID_FULL",
    "HF_REPO_TYPE",
    "SOURCE_URL",
    "PRESERVED_FIELDS",
    "ORACLE_FIELD",
    "ORACLE_FILENAMES",
    "download",
    "mask",
    "mask_row",
    "prepare",
]

# EOF
