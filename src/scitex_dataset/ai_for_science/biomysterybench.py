#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/biomysterybench.py

"""BioMysteryBench dataset preparation.

BioMysteryBench: biology-mystery problems with rubric-graded answers,
hosted on Hugging Face Hub by Anthropic. Two variants:

- ``Anthropic/BioMysteryBench-preview``  — 5 problems, ~11 MB, public.
- ``Anthropic/BioMysteryBench-full``     — 99 problems, ~159 GB, gated.

Pipeline (raw/masked contract — see :mod:`._base`):

1. ``download(...)`` — snapshot-download the preview tree (default) or
   the full set when ``download_full=True`` AND HF access is granted,
   storing the upstream tree *as-is* under ``raw_dir`` (answers and all).
   ``raw_dir`` is operator-private and never mounted into a capsule.
2. ``mask(...)`` — read ``raw_dir/problems.csv``, null the
   ``answer_rubric`` column, write JSONL to ``masked_dir/questions.jsonl``,
   then symlink the answer-free upstream content (e.g. ``data/*.zip``
   problem environments) into ``masked_dir``. ``masked_dir`` is the
   agent-visible, leak-safe view the harness mounts read-only.

NOTE on compute: the preview is small (~11 MB), the full set is multi-
GB and must run on SLURM. ``mask(...)`` is pure-Python on a small CSV
plus symlink creation and safe to run anywhere.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest

# Canonical benchmark identity.
BENCHMARK = "biomysterybench"
COHORT_ID = "biomysterybench"
COHORT_NAME = "BioMysteryBench"

HF_REPO_ID_PREVIEW = "Anthropic/BioMysteryBench-preview"
HF_REPO_ID_FULL = "Anthropic/BioMysteryBench-full"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID_PREVIEW}"

# Columns kept verbatim from problems.csv.
PRESERVED_FIELDS = ("id", "question", "allowed_domains", "human_solvable")
# Column whose value is replaced with None.
ORACLE_FIELD = "answer_rubric"

# Answer-bearing upstream files — kept in raw_dir, NEVER symlinked into
# the agent-visible masked view.
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


def _refresh_compat_symlinks(masked_dir: Path) -> None:
    for old_name, new_name in _COMPAT_SYMLINKS:
        link = masked_dir / old_name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(new_name)


def _link_safe_view(raw_dir: Path, masked_dir: Path, deny: set[str]) -> list[str]:
    """Symlink every answer-free top-level ``raw_dir`` entry into ``masked_dir``.

    Each link is RELATIVE (``../raw/<name>``) so the benchmark dir stays
    relocatable, and idempotent (an existing link/file at the target is
    unlinked first). Entries whose name is in ``deny`` (the oracle
    filenames) are skipped — the agent never sees them. Returns the list
    of created link paths (as strings).
    """
    created: list[str] = []
    for entry in sorted(raw_dir.iterdir()):
        # Skip the oracle deny-list and dot-prefixed HuggingFace/VCS
        # internals (``.cache``, ``.gitattributes``, …): neither is
        # benchmark content, and ``.cache`` is a latent leak-surface.
        if entry.name in deny or entry.name.startswith("."):
            continue
        link = masked_dir / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "raw" / entry.name)
        created.append(str(link))
    return created


def mask(
    *,
    raw_dir: Path,
    masked_dir: Path,
    **_,
) -> dict:
    """Read ``raw_dir/problems.csv``, build the masked view in ``masked_dir``.

    Writes ``questions.jsonl`` (JSONL, ``sort_keys=True``,
    ``ensure_ascii=False`` — symmetric with the other benchmarks) and
    symlinks the answer-free upstream content (problem environments)
    into ``masked_dir`` so the agent gets the data without the rubric.
    """
    src = raw_dir / "problems.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"biomysterybench: upstream source not found: {src}. "
            "Did you run download(...)?"
        )

    masked_dir.mkdir(parents=True, exist_ok=True)
    dst = masked_dir / "questions.jsonl"
    n = 0
    with (
        src.open("r", encoding="utf-8", newline="") as fh_in,
        dst.open("w", encoding="utf-8", newline="\n") as fh_out,
    ):
        reader = csv.DictReader(fh_in)
        for row in reader:
            fh_out.write(json.dumps(mask_row(row), sort_keys=True, ensure_ascii=False))
            fh_out.write("\n")
            n += 1

    symlinked = _link_safe_view(raw_dir, masked_dir, set(ORACLE_FILENAMES))
    _refresh_compat_symlinks(masked_dir)
    return {
        "output": str(dst),
        "n_records": n,
        "mask_fields": [ORACLE_FIELD],
        "symlinked": symlinked,
        "compat_symlinks": [str(masked_dir / old) for old, _ in _COMPAT_SYMLINKS],
    }


# ---------------------------------------------------------------------------
# Download — HF snapshot stored as-is under raw_dir.
# ---------------------------------------------------------------------------


def download(
    *,
    raw_dir: Path,
    download_full: bool = False,
    hf_token: str | None = None,
    max_workers: int = 4,
    **_,
) -> dict:
    """Pull the BMB HF snapshot into ``raw_dir`` exactly as-is.

    Preview snapshot (~11 MB, public) is always attempted. When
    ``download_full=True`` and HF access is granted the full set
    (~159 GB, gated) is pulled afterwards. ``huggingface_hub`` is
    required. No file is relocated — the answer-bearing artifacts stay
    in the operator-private ``raw_dir`` and leak-prevention happens at
    ``mask`` time (only ``masked_dir`` is ever mounted).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "biomysterybench.download requires huggingface_hub. "
            "Install with: pip install 'scitex-dataset[huggingface]'"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []

    snapshot_download(
        repo_id=HF_REPO_ID_PREVIEW,
        repo_type=HF_REPO_TYPE,
        local_dir=str(raw_dir),
        max_workers=max_workers,
        token=hf_token,
    )
    pulled.append(HF_REPO_ID_PREVIEW)

    if download_full:
        snapshot_download(
            repo_id=HF_REPO_ID_FULL,
            repo_type=HF_REPO_TYPE,
            local_dir=str(raw_dir),
            max_workers=max_workers,
            token=hf_token,
        )
        pulled.append(HF_REPO_ID_FULL)

    return {
        "raw_dir": str(raw_dir),
        "snapshots_pulled": pulled,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def prepare(
    *,
    paths: BenchmarkPaths | None = None,
    dataset_root: Path | str | None = None,
    version: str = "v0-unstamped",
    skip_download: bool = False,
    download_full: bool = False,
    **_,
) -> dict:
    """Run the full BioMysteryBench preparation pipeline.

    Pass ``download_full=True`` to attempt the gated 159 GB set after
    the preview. Pass ``skip_download=True`` to skip both HF pulls if
    the upstream CSV has already been hand-staged under ``raw_dir``.
    """
    if paths is None:
        paths = resolve_paths(BENCHMARK, dataset_root=dataset_root)

    out: dict = {"benchmark": BENCHMARK, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(
            raw_dir=paths.raw_dir,
            download_full=download_full,
        )
    out["mask"] = mask(raw_dir=paths.raw_dir, masked_dir=paths.masked_dir)

    manifest_path = write_manifest(
        manifest_dir=paths.manifest_dir,
        id=COHORT_ID,
        name=COHORT_NAME,
        version=version,
        source_url=SOURCE_URL,
        benchmark=BENCHMARK,
        tracked_paths=[Path(out["mask"]["output"])],
        tracked_root=paths.masked_dir,
        mask_seed="",
    )
    out["manifest"] = str(manifest_path)
    return out


__all__ = [
    "BENCHMARK",
    "COHORT_ID",
    "COHORT_NAME",
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
