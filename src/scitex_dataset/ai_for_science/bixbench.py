#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/bixbench.py

"""BixBench dataset preparation.

BixBench: 205 bioinformatics-analysis tasks across 54 capsules, hosted
on Hugging Face Hub as ``futurehouse/BixBench`` (public, no gate). The
HF snapshot bundles capsule code/data (``CapsuleData-<UUID>/`` +
``CapsuleNotebook-<UUID>/`` dirs) plus an answer-bearing
``BixBench.jsonl`` manifest.

Pipeline (raw → {for_solver, eval} contract — see :mod:`._base`):

1. ``download(...)`` — ``huggingface_hub.snapshot_download(...)`` into
   ``raw_dir`` exactly as-is (capsule dirs + the answer-bearing
   ``BixBench.jsonl``). ``raw_dir`` is operator-private and never
   mounted.
2. ``standardize(...)`` — read ``raw_dir/BixBench.jsonl``, split into the
   PER-CAPSULE agent-visible ``for_solver/`` view (one self-contained
   ``capsule-NNN/`` dir per native capsule — the EXTRACTED capsule archive
   in ``input/``, a ``task.jsonl`` of only that capsule's rows, the
   uniform submission schema/example, and a README — plus a root
   ``index.jsonl`` MAPPER of friendly_id ↔ native_id, no answers) and an
   operator-side ``eval/answers.jsonl`` + ``eval/evaluate.py``. Each
   record's ``data_folder`` (e.g. ``CapsuleFolder-<uuid>.zip``) is the
   capsule archive that gets extracted into its ``capsule-NNN/input/``. An
   agent binds exactly one ``capsule-NNN/`` dir; ``eval`` is operator-only.

NOTE on compute: the HF snapshot is ~16 GB across 67 files; SLURM-only
on shared compute. ``standardize(...)`` is pure-Python on the 205-record
JSONL (~285 KB) plus per-capsule archive extraction and safe to run
anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._base import BenchmarkPaths, resolve_paths
from ._manifest import write_manifest
from ._standardize import (
    render_evaluate_py,
    write_eval,
    write_for_solver_per_capsule,
)

# Canonical benchmark identity.
BENCHMARK = "bixbench"
COHORT_ID = "bixbench"
COHORT_NAME = "BixBench"
HF_REPO_ID = "futurehouse/BixBench"
HF_REPO_TYPE = "dataset"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Answer-bearing upstream manifest. Stays in raw_dir; NEVER symlinked
# into the agent-visible for_solver view.
ORACLE_MANIFEST_NAME = "BixBench.jsonl"

# Default scorer mode baked into eval/evaluate.py — BixBench answers are
# short free-text values, scored by normalized string equality.
DEFAULT_MODE = "string"


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
    """Read the oracle manifest, build the for_solver + eval views.

    Each upstream record becomes one leak-safe task (``{task_id,
    benchmark, prompt, data}``) + one answer (``{answer, ideal}``). Each
    task's ``data`` points at its ``data_folder`` archive (e.g.
    ``CapsuleFolder-<uuid>.zip``).

    ``for_solver`` is written in the PER-CAPSULE shape: one self-contained
    ``capsule-NNN/`` dir per native capsule (friendly id), each holding
    the EXTRACTED capsule archive in ``input/``, a ``task.jsonl`` of only
    that capsule's rows, the uniform submission schema/example, and a
    README — plus a root ``index.jsonl`` MAPPER (friendly_id ↔ native_id).
    An agent binds exactly one ``capsule-NNN/`` dir. Records with no
    ``data_folder`` have ``data: null`` and materialize no capsule.

    ``only`` (a friendly ``capsule-NNN`` id OR a native capsule id)
    materializes just that one capsule's dir; the mapper is always written
    in full. ``force`` re-extracts capsules already present.
    """
    src = raw_dir / ORACLE_MANIFEST_NAME
    if not src.is_file():
        raise FileNotFoundError(
            f"bixbench: upstream manifest not found: {src}. Did you run download(...)?"
        )

    tasks: list[dict] = []
    answers: list[dict] = []
    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = f"bixbench/{rec['short_id']}"
            data_folder = rec.get("data_folder")
            tasks.append(
                {
                    "task_id": task_id,
                    "benchmark": BENCHMARK,
                    "prompt": rec["question"],
                    "data": f"./{data_folder}" if data_folder else None,
                }
            )
            answers.append(
                {
                    "task_id": task_id,
                    "answer": {
                        "answer": rec.get("answer"),
                        "ideal": rec.get("ideal"),
                    },
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
    and leak-prevention happens at ``standardize`` time.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover — exercised via tests
        raise RuntimeError(
            "bixbench.download requires huggingface_hub. "
            "Install with: pip install 'scitex-dataset[huggingface]'"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    # snapshot_download already does per-file etag/sha skip natively, so
    # re-runs only re-pull files whose upstream content changed — no
    # extra integrity bookkeeping needed on our side (unlike corebench's
    # plain-HTTP capsule pull). We record the resolved snapshot set below.
    resolved = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(raw_dir),
        max_workers=max_workers,
        token=hf_token,
    )
    return {
        "raw_dir": str(raw_dir),
        "snapshots_pulled": [HF_REPO_ID],
        "resolved": str(resolved),
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
    "HF_REPO_ID",
    "HF_REPO_TYPE",
    "SOURCE_URL",
    "DEFAULT_MODE",
    "ORACLE_MANIFEST_NAME",
    "download",
    "standardize",
    "prepare",
]

# EOF
