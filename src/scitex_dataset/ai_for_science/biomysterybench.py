#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/biomysterybench.py

"""BioMysteryBench dataset preparation.

BioMysteryBench: biology-mystery problems with rubric-graded answers,
hosted on Hugging Face Hub by Anthropic. Two variants:

- ``Anthropic/BioMysteryBench-preview``  — 5 problems, ~11 MB, public.
- ``Anthropic/BioMysteryBench-full``     — 99 problems, ~159 GB, gated.

Pipeline (raw → {for_solver, eval} contract — see :mod:`._base`):

1. ``download(...)`` — snapshot-download the preview tree (default) or
   the full set when ``download_full=True`` AND HF access is granted,
   storing the upstream tree *as-is* under ``raw_dir`` (answers and all).
   ``raw_dir`` is operator-private and never mounted into a capsule.
2. ``standardize(...)`` — read ``raw_dir/problems.csv``, split into the
   PER-CAPSULE agent-visible ``for_solver/`` view (one self-contained
   ``capsule-NNN/`` dir per problem that ships an environment — the
   EXTRACTED ``data/<id>.zip`` in ``input/``, a ``task.jsonl`` of only
   that problem's row, the uniform submission schema/example, and a
   README — plus a root ``index.jsonl`` MAPPER of friendly_id ↔ native_id,
   no rubric) and an operator-side ``eval/answers.jsonl`` (carrying the
   rubric) + ``eval/evaluate.py``. A problem's ``data`` points at
   ``data/<id>.zip`` when present (extracted into its ``capsule-NNN/
   input/``), else ``null`` (no capsule materialized). An agent binds
   exactly one ``capsule-NNN/`` dir; ``eval`` is operator-only.

NOTE on compute: the preview is small (~11 MB), the full set is multi-
GB and must run on SLURM. ``standardize(...)`` is pure-Python on a small
CSV plus per-capsule archive extraction and safe to run anywhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest
from ._standardize import (
    render_evaluate_py,
    write_eval,
    write_for_solver_per_capsule,
)

# Canonical benchmark identity.
BENCHMARK = "biomysterybench"
COHORT_ID = "biomysterybench"
COHORT_NAME = "BioMysteryBench"

HF_REPO_ID_PREVIEW = "Anthropic/BioMysteryBench-preview"
HF_REPO_ID_FULL = "Anthropic/BioMysteryBench-full"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID_PREVIEW}"

# Column carrying the rubric answer; moved to eval/, never in for_solver.
ORACLE_FIELD = "answer_rubric"

# Answer-bearing upstream files — kept in raw_dir, NEVER symlinked into
# the agent-visible for_solver view.
ORACLE_FILENAMES = ("problems.csv", "problems.parquet", "README.md")

# Default scorer mode baked into eval/evaluate.py — BMB answers are
# rubric-graded (not auto-scorable), so evaluate.py marks them for
# manual grading rather than computing a numeric/string score.
DEFAULT_MODE = "rubric"

# Subdir under raw_dir holding per-problem zipped environments.
_DATA_SUBDIR = "data"


# ---------------------------------------------------------------------------
# Standardize — pure-Python, network-free.
# ---------------------------------------------------------------------------


def standardize(
    *,
    raw_dir: Path,
    for_solver_dir: Path,
    eval_dir: Path,
    only: str | None = None,
    force: bool = False,
    **_,
) -> dict:
    """Read ``raw_dir/problems.csv``, build the for_solver + eval views.

    Each CSV row becomes one leak-safe task (``{task_id, benchmark,
    prompt, data}``) + one answer (``{rubric}``). A task's ``data``
    points at ``data/<id>.zip`` when that environment exists under
    ``raw/data``, else ``null``. The rubric stays in operator-only
    ``eval/answers.jsonl``.

    ``for_solver`` is written in the PER-CAPSULE shape: one self-contained
    ``capsule-NNN/`` dir per problem that ships a ``data/<id>.zip``
    environment (friendly id), each holding the EXTRACTED archive in
    ``input/``, a ``task.jsonl`` of only that problem's row, the uniform
    submission schema/example, and a README — plus a root ``index.jsonl``
    MAPPER (friendly_id ↔ native_id). An agent binds exactly one
    ``capsule-NNN/`` dir. A problem with ``data: null`` (no environment)
    materializes no capsule.

    ``only`` (a friendly ``capsule-NNN`` id OR a native capsule id)
    materializes just that one capsule's dir; the mapper is always written
    in full. ``force`` re-extracts capsules already present.
    """
    src = raw_dir / "problems.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"biomysterybench: upstream source not found: {src}. "
            "Did you run download(...)?"
        )

    data_dir = raw_dir / _DATA_SUBDIR
    tasks: list[dict] = []
    answers: list[dict] = []
    with src.open("r", encoding="utf-8", newline="") as fh_in:
        reader = csv.DictReader(fh_in)
        for row in reader:
            rid = row["id"]
            task_id = f"biomysterybench/{rid}"
            zip_path = data_dir / f"{rid}.zip"
            tasks.append(
                {
                    "task_id": task_id,
                    "benchmark": BENCHMARK,
                    "prompt": row["question"],
                    "data": f"./data/{rid}.zip" if zip_path.exists() else None,
                }
            )
            answers.append(
                {
                    "task_id": task_id,
                    "answer": {"rubric": row.get(ORACLE_FIELD)},
                }
            )

    fs = write_for_solver_per_capsule(
        for_solver_dir=for_solver_dir,
        tasks=tasks,
        raw_dir=raw_dir,
        only=only,
        force=force,
    )
    ev = write_eval(
        eval_dir=eval_dir,
        answers=answers,
        evaluate_py_source=render_evaluate_py(DEFAULT_MODE),
    )
    return {
        "for_solver": fs,
        "eval": ev,
        "n_tasks": len(tasks),
        "default_mode": DEFAULT_MODE,
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
    ``standardize`` time (only ``for_solver`` is ever mounted).
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

    # snapshot_download already does per-file etag/sha skip natively, so
    # re-runs only re-pull files whose upstream content changed — no
    # extra integrity bookkeeping needed on our side.
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
    out["standardize"] = standardize(
        raw_dir=paths.raw_dir,
        for_solver_dir=paths.for_solver_dir,
        eval_dir=paths.eval_dir,
    )

    manifest_path = write_manifest(
        manifest_dir=paths.manifest_dir,
        id=COHORT_ID,
        name=COHORT_NAME,
        version=version,
        source_url=SOURCE_URL,
        benchmark=BENCHMARK,
        tracked_paths=[Path(out["standardize"]["for_solver"]["index"])],
        tracked_root=paths.for_solver_dir,
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
    "DEFAULT_MODE",
    "ORACLE_FIELD",
    "ORACLE_FILENAMES",
    "download",
    "standardize",
    "prepare",
]

# EOF
