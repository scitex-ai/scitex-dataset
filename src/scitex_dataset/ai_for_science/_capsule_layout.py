#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_capsule_layout.py

"""Per-capsule layout: friendly ids, mapper, archive extraction, README.

Split from ``_standardize``; the writer there orchestrates these pieces.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from ._identifiers import LOCAL_ENV_IMAGE

# Archive suffixes recognised for extraction, longest first so the
# compound ``.tar.gz`` wins over a bare ``.gz`` when stripping the id.
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

# Files copied/written into every per-capsule dir.
_TASK_FILENAME = "task.jsonl"
_INPUT_SUBDIR = "input"
_INDEX_FILENAME = "index.jsonl"
_README_FILENAME = "README.md"
_SCHEMA_FILENAME = "submission.schema.json"
_EXAMPLE_FILENAME = "submission.example.json"

# Per-capsule friendly-id directory prefix (``capsule-NNN``). The
# Solver capsules are separate from operator-root mapper, identity ledger and
# optional inventory. Legacy roots refuse BEFORE any purge: they are retained
# for historical evidence and must not be rebound into a solver capsule.
_CAPSULE_DIR_PREFIX = "capsule-"
_IDENTITY_FILENAME = ".materialization-identity.json"
_OPERATOR_ROOT_FILES = {_INDEX_FILENAME, _IDENTITY_FILENAME, "inventory.json"}


def friendly_capsule_id(index: int) -> str:
    """Return the deterministic friendly dir name for a 0-based position.

    Friendly ids are ``capsule-NNN`` zero-padded to 3 digits and 1-based
    (``index=0`` → ``capsule-001``), assigned over the native capsule ids
    sorted ascending PER BENCHMARK.
    """
    return f"capsule-{index + 1:03d}"


def _native_capsule_id(data: str | None) -> str | None:
    """Derive the native capsule id from a task's ``data`` archive path.

    ``data`` points at the capsule's archive, e.g.
    ``"./capsules/capsule-0201225.tar.gz"`` →  ``"capsule-0201225"``.
    The leading directory and any recognised archive suffix are stripped.
    Returns ``None`` when ``data`` is falsy (task has no bound archive).
    """
    if not data:
        return None
    name = Path(data).name
    for suffix in _ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_capsule_index(tasks: list[dict]) -> list[dict]:
    """Group ``tasks`` by native capsule and assign friendly ids.

    Tasks are grouped by the native capsule id derived from each row's
    ``data`` archive path; the distinct native ids are sorted ascending
    and given ``capsule-NNN`` friendly ids in that order. Returns one
    mapper row per capsule, in friendly-id order::

        {"friendly_id", "native_id", "benchmark", "task_ids", "dir"}

    This is computed from the FULL task list (independent of any ``only``
    filter) so the mapper is always complete. Tasks whose ``data`` yields
    no capsule id are skipped (they have nothing to materialize).
    """
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        native = _native_capsule_id(task.get("data"))
        if native is None:
            continue
        grouped.setdefault(native, []).append(task)

    index: list[dict] = []
    for position, native in enumerate(sorted(grouped)):
        friendly = friendly_capsule_id(position)
        rows = grouped[native]
        index.append(
            {
                "friendly_id": friendly,
                "native_id": native,
                "benchmark": rows[0].get("benchmark"),
                "task_ids": [r["task_id"] for r in rows],
                "dir": friendly,
            }
        )
    return index


def _resolve_only(only: str, index: list[dict]) -> dict:
    """Resolve an ``--only`` selector (friendly OR native id) to its row.

    Raises ``KeyError`` if it matches neither a ``friendly_id`` nor a
    ``native_id`` in the mapper.
    """
    for row in index:
        if only in (row["friendly_id"], row["native_id"]):
            return row
    raise KeyError(
        f"--only {only!r} matched no capsule (neither a friendly_id "
        f"capsule-NNN nor a native capsule id) in the {len(index)}-capsule index."
    )


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract ``archive_path`` into ``dest_dir`` (stdlib, by extension).

    ``.tar.gz`` / ``.tgz`` / ``.tar`` go through :mod:`tarfile`; ``.zip``
    through :mod:`zipfile`. ``dest_dir`` is created fresh by the caller.
    Raises ``FileNotFoundError`` if the archive is missing (fail loud:
    a selected capsule with no archive on disk is an operator error),
    and ``ValueError`` for an unrecognised suffix.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"capsule archive not found (or is a dangling symlink): {archive_path}"
        )
    name = archive_path.name
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(
            f"unrecognised archive suffix for {archive_path} "
            f"(expected one of {_ARCHIVE_SUFFIXES})."
        )


def _flatten_single_top_dir(input_dir: Path) -> None:
    """De-nest ``input_dir`` when the archive wrapped everything in one dir.

    Many capsule tarballs (e.g. CoreBench) contain a SINGLE top-level
    directory named after the capsule (``capsule-0220918/``), so a naive
    extract yields ``input/capsule-0220918/code…`` instead of
    ``input/code…`` — and the task's ``data: "./input"`` pointer then
    misses by one level. If ``input_dir`` holds EXACTLY one entry and it
    is a directory, this lifts that directory's children up into
    ``input_dir`` and removes the now-empty wrapper. The general case is
    handled conservatively: a multi-entry archive (or a lone top-level
    FILE) is left untouched.
    """
    entries = list(input_dir.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    wrapper = entries[0]
    # Two-phase to dodge a name collision with the wrapper itself: stage
    # to a sibling temp dir, then move children up and drop the wrapper.
    staging = input_dir.parent / f"_flatten_{input_dir.name}"
    if staging.exists():
        _rmtree(staging)
    wrapper.rename(staging)
    for child in staging.iterdir():
        child.rename(input_dir / child.name)
    staging.rmdir()


def _purge_legacy_root_artifacts(for_solver_dir: Path) -> list[str]:
    """Keep capsule directories and operator-root administrative files.

    The writer first refuses legacy artifacts without changing them. This
    internal cleanup is reached ONLY after identity qualification. Every other
    root entry re-exposes capsules/tasks beyond the single capsule an agent
    binds, so it is removed regardless of name:

    - the old flat layout's global ``tasks.jsonl`` / ``submission.*``;
    - benchmark-specific problem-data symlinks the old
      :func:`write_for_solver` created at the root (CoreBench ``capsules``,
      BixBench ``CapsuleFolder-*.zip``, BioMysteryBench ``data``).

    A ``capsule-NNN`` REAL directory, mapper, private identity ledger and
    optional operator inventory are preserved. Returns the removed
    paths as strings.
    """
    if not for_solver_dir.is_dir():
        return []
    removed: list[str] = []
    for entry in sorted(for_solver_dir.iterdir()):
        if entry.name in _OPERATOR_ROOT_FILES:
            continue
        is_capsule_dir = (
            entry.is_dir()
            and not entry.is_symlink()
            and entry.name.startswith(_CAPSULE_DIR_PREFIX)
            and entry.name[len(_CAPSULE_DIR_PREFIX) :].isdigit()
        )
        if is_capsule_dir:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            _rmtree(entry)
        removed.append(str(entry))
    return removed


def _render_capsule_readme(
    *,
    friendly_id: str,
    benchmark: str | None,
    task_ids: list[str],
    has_environment: bool = False,
) -> str:
    """Return plain-language submission instructions for one capsule.

    Names the capsule by its FRIENDLY id, lists the exact ``task_id``(s)
    to answer, states the uniform JSON-array submission contract, and
    points at ``submission.example.json`` + the extracted ``input/`` dir.
    """
    bench = benchmark or "this benchmark"
    env_note = (
        f"\nThe software environment is defined by `{_INPUT_SUBDIR}/environment/Dockerfile`.\n"
        f"References to the published image now name the local tag "
        f"`{LOCAL_ENV_IMAGE}`;\nbuild it with "
        f"`docker build -t {LOCAL_ENV_IMAGE} {_INPUT_SUBDIR}/environment`.\n"
        if has_environment
        else ""
    )
    id_lines = "\n".join(f"- `{tid}`" for tid in task_ids)
    return f"""# {friendly_id}

This directory is one self-contained {bench} task capsule. Everything
you need is inside it; no other capsule is referenced.

## What to do

1. The problem's code and data are EXTRACTED under `{_INPUT_SUBDIR}/`.
   Read `{_INPUT_SUBDIR}/` and work the task described in `{_TASK_FILENAME}`.
2. Answer the following task id(s):

{id_lines}

## How to submit

Write a single JSON file: an ARRAY of objects, one per task id above,
each shaped `{{"task_id": "<id>", "answer": <your answer>}}` — exactly
the contract in `{_SCHEMA_FILENAME}`. A ready-to-edit template with the
real task id(s) filled in is in `{_EXAMPLE_FILENAME}`; replace each
`"<your answer here>"` with your answer and keep the `task_id` values
unchanged.

## Redactions

Pointers to the originally published results (hosting-platform links,
published environment images, digests, DOIs, hosted result files) were
replaced by markers such as `[redacted-url]`, `[redacted-doi]` and
`[redacted-identifier]`. Reproduce the results from the code and data in
`{_INPUT_SUBDIR}/`; dependency lists and run commands are unchanged.
{env_note}"""


def _rmtree(path: Path) -> None:
    """Recursively remove a directory tree (stdlib ``shutil`` lazy import)."""
    import shutil

    shutil.rmtree(path)


# EOF
