#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/bixbench.py

"""BixBench dataset preparation.

BixBench: 205 bioinformatics-analysis tasks across 54 capsules, hosted
on Hugging Face Hub as ``futurehouse/BixBench`` (public, no gate). The
HF snapshot bundles capsule code/data (``CapsuleData-<UUID>/`` +
``CapsuleNotebook-<UUID>/`` dirs) plus an answer-bearing
``BixBench.jsonl`` manifest.

Pipeline (raw/masked contract — see :mod:`._base`):

1. ``download(...)`` — ``huggingface_hub.snapshot_download(...)`` into
   ``raw_dir`` exactly as-is (capsule dirs + the answer-bearing
   ``BixBench.jsonl``). ``raw_dir`` is operator-private and never
   mounted.
2. ``mask(...)`` — read ``raw_dir/BixBench.jsonl``, null the oracle
   fields (``answer``, ``ideal``, ``result``, ``distractors``,
   ``paper``), write ``masked_dir/questions.jsonl``, then symlink the
   answer-free capsule content into ``masked_dir``. Preserves
   ``hypothesis`` (paper-level context, not an answer spoiler) and
   ``canary`` (leakage-detection sentinel). ``masked_dir`` is the
   agent-visible, leak-safe view mounted read-only.

NOTE on compute: the HF snapshot is ~16 GB across 67 files; SLURM-only
on shared compute. ``mask(...)`` is pure-Python on the 205-record
JSONL (~285 KB) plus symlink creation and safe to run anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest

# Canonical benchmark identity.
BENCHMARK = "bixbench"
COHORT_ID = "bixbench"
COHORT_NAME = "BixBench"
HF_REPO_ID = "futurehouse/BixBench"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Answer-bearing upstream manifest. Stays in raw_dir; NEVER symlinked
# into the agent-visible masked view.
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


def _refresh_compat_symlinks(masked_dir: Path) -> None:
    for old_name, new_name in _COMPAT_SYMLINKS:
        link = masked_dir / old_name
        if link.is_symlink() or link.exists():
            link.unlink()
        # Relative target keeps the symlink host-portable.
        link.symlink_to(new_name)


def _link_safe_view(raw_dir: Path, masked_dir: Path, deny: set[str]) -> list[str]:
    """Symlink every answer-free top-level ``raw_dir`` entry into ``masked_dir``.

    Each link is RELATIVE (``../raw/<name>``) so the benchmark dir stays
    relocatable, and idempotent. Entries whose name is in ``deny`` (the
    oracle filenames) are skipped. Returns the created link paths.
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
    """Read the oracle manifest, build the masked view in ``masked_dir``.

    Output is JSONL: one record per line, ``sort_keys=True`` and
    ``ensure_ascii=False`` for deterministic byte output across runs.
    Symlinks the answer-free capsule content into ``masked_dir`` and
    recreates the legacy ``BixBench_masked.jsonl`` backward-compat link.
    """
    src = raw_dir / ORACLE_MANIFEST_NAME
    if not src.is_file():
        raise FileNotFoundError(
            f"bixbench: upstream manifest not found: {src}. Did you run download(...)?"
        )

    masked_dir.mkdir(parents=True, exist_ok=True)
    dst = masked_dir / "questions.jsonl"
    n = 0
    with (
        src.open("r", encoding="utf-8") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            masked = mask_record(rec)
            fout.write(json.dumps(masked, sort_keys=True, ensure_ascii=False))
            fout.write("\n")
            n += 1

    symlinked = _link_safe_view(raw_dir, masked_dir, {ORACLE_MANIFEST_NAME})
    _refresh_compat_symlinks(masked_dir)
    return {
        "output": str(dst),
        "n_records": n,
        "mask_fields": list(MASK_FIELDS),
        "symlinked": symlinked,
        "compat_symlinks": [str(masked_dir / old) for old, _ in _COMPAT_SYMLINKS],
    }


# ---------------------------------------------------------------------------
# Download — HF snapshot stored as-is under raw_dir.
# ---------------------------------------------------------------------------


def download(
    *,
    raw_dir: Path,
    hf_token: str | None = None,
    max_workers: int = 4,
    **_,
) -> dict:
    """Pull the BixBench HF snapshot into ``raw_dir`` exactly as-is.

    HEAVY: ~16 GB across ~67 files; SLURM-only on shared compute.
    ``huggingface_hub`` is an optional extra (``scitex-dataset[mcp]``
    doesn't include it) — install ``scitex-dataset[huggingface]`` or
    ``scitex-dataset[all]`` first. No file is relocated — the
    answer-bearing manifest stays in the operator-private ``raw_dir``
    and leak-prevention happens at ``mask`` time.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover — exercised via tests
        raise RuntimeError(
            "bixbench.download requires huggingface_hub. "
            "Install with: pip install 'scitex-dataset[huggingface]'"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(raw_dir),
        max_workers=max_workers,
        token=hf_token,
    )
    return {
        "raw_dir": str(raw_dir),
        "snapshots_pulled": [HF_REPO_ID],
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
    **_,
) -> dict:
    """Run the full BixBench preparation pipeline.

    Set ``skip_download=True`` to skip the HF snapshot pull (useful if
    the upstream manifest has already been hand-staged under ``raw_dir``).
    """
    if paths is None:
        paths = resolve_paths(BENCHMARK, dataset_root=dataset_root)

    out: dict = {"benchmark": BENCHMARK, "paths": paths.as_dict()}
    if not skip_download:
        out["download"] = download(raw_dir=paths.raw_dir)
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
