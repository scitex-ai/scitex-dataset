#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_standardize.py

"""Shared ``standardize`` helpers — raw oracle → {for_solver, eval}.

Every AI-for-science benchmark splits its operator-private ``raw/``
oracle (which *has* answers) into two derived views with a UNIFORM
on-disk contract, so the experiment harness treats all benchmarks
identically:

- ``for_solver/`` — the AGENT view (leak-safe). Two shapes are
  supported:

  * the FLAT shape (:func:`write_for_solver`) — a global ``tasks.jsonl``
    of ``{task_id, benchmark, prompt, data}`` records (NO answers), a
    ``submission.schema.json``, a one-row ``submission.example.json``,
    and relative symlinks to the answer-free problem data under
    ``raw/``;
  * the PER-CAPSULE shape (:func:`write_for_solver_per_capsule`) — one
    self-contained ``capsule-NNN/`` dir per native capsule (FRIENDLY id,
    human-communicable) holding the EXTRACTED problem archive in
    ``input/``, a ``task.jsonl`` of ONLY that capsule's rows (each
    ``data`` rewritten to ``./input``), a copied
    ``submission.schema.json``, a ``submission.example.json`` pre-filled
    with this capsule's real ``task_id``(s), and a plain-language
    ``README.md``. A root ``index.jsonl`` MAPPER (operator-facing, never
    mounted) records every friendly_id → native_id assignment. An agent
    binds EXACTLY one ``capsule-NNN/`` dir so no sibling capsule (and no
    answer) is ever reachable.
- ``eval/`` — the OPERATOR view (never mounted). An ``answers.jsonl``
  keyed by the same ``task_id`` as ``tasks.jsonl`` plus a stdlib-only
  ``evaluate.py`` CLI scorer. The scorer is answer-free logic — the
  answers are passed in as a file argument, never baked into the code.

Per-benchmark modules build the ``tasks`` + ``answers`` lists with their
own mappers, then call :func:`write_for_solver` and :func:`write_eval`.
The single shared :data:`EVALUATE_PY` template implements a uniform
scorer; each benchmark bakes its default ``--mode`` into the rendered
copy via the ``{default_mode}`` placeholder.

This module is the facade + writers. Focused helpers live in
:mod:`._evaluate_template` (scorer template), :mod:`._capsule_layout`
(friendly ids, mapper, archive extraction, README), :mod:`._leak_strip`
(answer-leak strips/redaction), :mod:`._identifiers` (hosting-identifier
redaction) and :mod:`._materialization_identity` (cache identity ledger);
their names are re-exported here for existing callers.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ._capsule_layout import (  # noqa: F401 - re-exported
    _ARCHIVE_SUFFIXES,
    _CAPSULE_DIR_PREFIX,
    _EXAMPLE_FILENAME,
    _IDENTITY_FILENAME,
    _INDEX_FILENAME,
    _INPUT_SUBDIR,
    _OPERATOR_ROOT_FILES,
    _README_FILENAME,
    _SCHEMA_FILENAME,
    _TASK_FILENAME,
    _extract_archive,
    _flatten_single_top_dir,
    _native_capsule_id,
    _purge_legacy_root_artifacts,
    _render_capsule_readme,
    _resolve_only,
    _rmtree,
    build_capsule_index,
    friendly_capsule_id,
)
from ._evaluate_template import EVALUATE_PY, render_evaluate_py  # noqa: F401
from ._identifiers import (  # noqa: F401 - re-exported
    HostingIdentifierLeakError,
    assert_no_hosting_identifiers,
    mask_hosting_identifiers,
)
from ._leak_strip import (  # noqa: F401 - re-exported
    _SCAN_EXTS,
    _VALUE_REDACTION,
    LEAK_DIRS,
    AnswerLeakError,
    InvalidReferenceSourceError,
    _answer_targets,
    _assert_no_value_leak,
    _mask_value_leaks,
    _redact_text,
    _strip_leak_dirs,
    _strip_leak_files,
    _strip_notebook_outputs,
)
from ._materialization_identity import (  # noqa: F401 - re-exported
    _MATERIALIZATION_CONTRACT,
    StaleMaterializationError,
    _capsule_identifiers,
    _capsule_reference_values,
    _json_identity,
    _materialized_identity,
    _qualify_materializations,
    _sha256_file,
)
from ._neutral_paths import (  # noqa: F401 - re-exported
    IdentifierPathLeakError,
    assert_no_identifier_paths,
    neutralize_identifier_paths,
)

# The uniform submission contract: an array of objects, each carrying the
# ``task_id`` it answers plus the agent's ``answer`` (any JSON value).
UNIFORM_SUBMISSION_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ai-for-science uniform submission",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "answer": {},
        },
        "required": ["task_id", "answer"],
        "additionalProperties": True,
    },
}

# Uniform keys every for_solver task row carries — used by callers and
# tests to assert the leak-safe schema is exactly these (no answer-
# bearing keys leak through).
TASK_KEYS = ("task_id", "benchmark", "prompt", "data")
# Public optional declaration, supplied by a task adapter, never by an oracle.
ANSWER_TYPES = {"json", "number", "integer", "string", "boolean", "list", "object"}


def submission_schema_for_tasks(tasks: list[dict]) -> dict:
    """Build a capsule schema from public task declarations only.

    An absent ``answer_type`` retains arbitrary JSON. Null/reason handling
    remains the submission validator's contract; null remains representable.
    """
    schema = json.loads(json.dumps(UNIFORM_SUBMISSION_SCHEMA))
    conditions = []
    seen = set()
    for task in tasks:
        tid = task.get("task_id")
        if not isinstance(tid, str) or not tid.strip() or tid in seen:
            raise ValueError("assigned task IDs must be nonempty and unique")
        seen.add(tid)
        answer_type = task.get("answer_type", "json")
        if not isinstance(answer_type, str) or answer_type not in ANSWER_TYPES:
            raise ValueError("unsupported public answer_type declaration")
        if answer_type == "json":
            continue
        json_type = {"list": "array", "object": "object"}.get(answer_type, answer_type)
        conditions.append(
            {
                "if": {
                    "properties": {"task_id": {"const": task["task_id"]}},
                    "required": ["task_id"],
                },
                "then": {"properties": {"answer": {"type": [json_type, "null"]}}},
            }
        )
    if conditions:
        schema["items"]["allOf"] = conditions
    return schema


# ---------------------------------------------------------------------------
# Writers — shared by every benchmark's standardize().
# ---------------------------------------------------------------------------


def _link_problem_data(
    for_solver_dir: Path, raw_dir: Path, data_links: list[str]
) -> list[str]:
    """Create relative ``../raw/<name>`` symlinks for each problem-data name.

    Idempotent (an existing link/file at the target is unlinked first).
    Skips dot-prefixed names (VCS / HF internals) and names with no
    matching ``raw_dir`` entry. Returns the created link paths as strings.
    """
    created: list[str] = []
    for name in data_links:
        if name.startswith("."):
            continue
        if not (raw_dir / name).exists():
            continue
        link = for_solver_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "raw" / name)
        created.append(str(link))
    return created


def write_for_solver(
    *,
    for_solver_dir: Path,
    tasks: list[dict],
    raw_dir: Path,
    data_links: list[str],
) -> dict:
    """Write the agent-visible ``for_solver/`` view.

    Emits ``tasks.jsonl`` (one task per line, ``sort_keys=True``,
    ``ensure_ascii=False`` for deterministic bytes), the uniform
    ``submission.schema.json`` and a ``submission.example.json`` built
    from the first task's ``task_id`` with a placeholder answer, then
    symlinks the answer-free problem data named in ``data_links``.
    """
    schema = submission_schema_for_tasks(tasks)
    for_solver_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = for_solver_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    schema_path = for_solver_dir / "submission.schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    example_id = tasks[0]["task_id"] if tasks else "<task_id>"
    example_path = for_solver_dir / "submission.example.json"
    example_path.write_text(
        json.dumps(
            [{"task_id": example_id, "answer": "<your answer here>"}],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    symlinked = _link_problem_data(for_solver_dir, raw_dir, data_links)
    return {
        "tasks": str(tasks_path),
        "n_tasks": len(tasks),
        "schema": str(schema_path),
        "example": str(example_path),
        "symlinked": symlinked,
    }


# ---------------------------------------------------------------------------
# Per-capsule materializer — one self-contained, friendly-id'd dir per
# native capsule, plus a root index.jsonl MAPPER. This is the shape an
# agent binds (EXACTLY one capsule-NNN/ dir, never the whole tree).
# ---------------------------------------------------------------------------


def write_for_solver_per_capsule(
    *,
    for_solver_dir: Path,
    tasks: list[dict],
    raw_dir: Path,
    only: str | None = None,
    force: bool = False,
    answer_values: dict[str, object] | None = None,
    reference_values: dict[str, list[object]] | None = None,
    source_identity: object = None,
    identifiers: dict[str, list[str]] | None = None,
    path_patterns: list[str] | None = None,
    eval_dir: Path | None = None,
) -> dict:
    """Write the per-capsule, friendly-id'd ``for_solver/`` view + mapper.

    Groups ``tasks`` by native capsule (via each row's ``data`` archive
    path), assigns deterministic ``capsule-NNN`` friendly ids over the
    sorted native ids, and writes the operator-facing mapper
    ``for_solver/index.jsonl`` (one row per capsule — ALWAYS the full set,
    regardless of ``only``). Then, for each materialized capsule, creates
    a self-contained ``for_solver/capsule-NNN/`` dir holding:

    - ``input/``                  — the raw archive EXTRACTED here, then
                                de-nested if it wrapped everything in a
                                single top-level dir (so ``input/`` holds
                                ``code/``, ``data/``, … directly), and with
                                the answer-bearing :data:`LEAK_DIRS` (e.g.
                                ``results/``) stripped so the agent cannot
                                read the authors' original outputs;
    - ``task.jsonl``              — ONLY this capsule's task rows, each
                                row's ``data`` rewritten to ``"./input"``;
    - ``submission.schema.json``  — the uniform submission schema;
    - ``submission.example.json`` — pre-filled with THIS capsule's real
                                task_id(s) + placeholder answers;
    - ``README.md``               — plain-language instructions.

    Before anything is written, all existing materializations must match
    their operator-private root identity ledger (tasks/native/archive/source
    and actual cached files). Legacy or changed content refuses even with
    ``force=True``; use a fresh destination and preserve old outputs.

    ``only`` (friendly id ``capsule-NNN`` OR a native capsule id)
    restricts materialization to a single capsule's dir; the mapper is
    still written in full. Default materializes every capsule.

    Only identity-qualified existing capsules may be skipped or re-extracted.
    ``reference_values`` supplies per-run samples to the EXISTING leak guard;
    a list-valued answer remains one sample. No masking policy is changed.
    Nonfinite/unrepresentable masking targets refuse before outputs change.
    Optional ``source_identity`` is retained only in the operator root ledger;
    callers must qualify its actual upstream revision separately.

    Path names are neutralised too (#111): every ``input/`` path component
    carrying an identifier (any UUID, the capsule's exact identifiers, their
    long digit cores, the adapter's ``path_patterns`` regexes) is renamed to
    ``folder_NN`` / ``notebook_NN.ipynb`` / ``file_NN.<ext>`` with references
    in text files and symlinks rewritten, and the path guard then refuses any
    surviving identifier in a capsule path. The old→new mapping is written to
    ``eval_dir/path_renames.json`` (operator side, never under for_solver) and
    returned as ``path_renames``.

    ``raw_dir`` is the operator-private snapshot root; each capsule's
    archive is resolved as ``raw_dir / <data-relative-to-for_solver>``.
    """
    index = build_capsule_index(tasks)
    index_path = for_solver_dir / _INDEX_FILENAME

    if only is not None:
        selected = [_resolve_only(only, index)]
    else:
        selected = index

    # Re-group tasks by native id once so each capsule writes only its own.
    by_native: dict[str, list[dict]] = {}
    for task in tasks:
        native = _native_capsule_id(task.get("data"))
        if native is not None:
            by_native.setdefault(native, []).append(task)

    # Validate public declarations/identity before mapper, purge or extraction.
    submission_schema_for_tasks(tasks)
    ledger, identities = _qualify_materializations(
        for_solver_dir,
        index,
        selected,
        by_native,
        raw_dir,
        source_identity,
        answer_values,
        reference_values,
        identifiers,
        path_patterns,
    )
    for_solver_dir.mkdir(parents=True, exist_ok=True)
    purged_legacy = _purge_legacy_root_artifacts(for_solver_dir)

    materialized: list[str] = []
    skipped: list[str] = []
    stripped_leaks: list[str] = []
    stripped_files: list[str] = []
    masked_values: list[str] = []
    cleared_notebooks: list[str] = []
    masked_identifiers: list[str] = []
    identifier_classes: dict[str, int] = {}
    identifier_unscanned: list[str] = []
    path_renames: dict[str, dict] = {}
    patterns = list(path_patterns or [])
    for row in selected:
        native = row["native_id"]
        capsule_dir = for_solver_dir / row["dir"]
        input_dir = capsule_dir / _INPUT_SUBDIR
        rows = by_native[native]

        # The archive path is shared across this capsule's rows.
        archive_rel = rows[0]["data"]
        archive_path = (raw_dir / archive_rel).resolve()

        if input_dir.exists() and not force:
            skipped.append(str(capsule_dir))
            continue
        if force and input_dir.exists():
            _rmtree(input_dir)

        capsule_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, input_dir)
        # De-nest a single wrapping top-level dir so input/ holds code/,
        # data/, … directly, making the ``data: "./input"`` pointer exact.
        _flatten_single_top_dir(input_dir)
        # Strip the answer-leak AFTER flattening so it catches input/results/
        # (the authors' original outputs) regardless of the archive shape.
        stripped_leaks.extend(_strip_leak_dirs(input_dir))
        # Loose running-log files outside the leak dirs (e.g. data/*_log.csv).
        stripped_files.extend(_strip_leak_files(input_dir))
        # Clear executed-notebook output cells (e.g. BixBench's
        # ``*_executed.ipynb`` leaks the answer in its outputs) — keeps the
        # notebook CODE as scaffold, removes the computed answers. No-op for
        # capsules without notebooks.
        cleared_notebooks.extend(_strip_notebook_outputs(input_dir))
        # Neutralise identifier-bearing PATH NAMES before any content masking,
        # so references are rewritten to the new names while the old names are
        # still literally present (content masking would otherwise turn them
        # into ``[redacted-identifier]`` and break the reference).
        exact = _capsule_identifiers(native, identifiers)
        renames = neutralize_identifier_paths(input_dir, exact, patterns)
        path_renames[row["dir"]] = {"native_id": native, "renames": renames}
        # Redact any distinctive answer value baked into a KEPT file (e.g. an
        # extensionless code/evaluation dump) the structural strips can't catch,
        # then fail loud if anything still survives (the assert is the backstop).
        if answer_values or reference_values:
            cap_values = _capsule_reference_values(
                rows, answer_values, reference_values
            )
            masked_values.extend(_mask_value_leaks(input_dir, cap_values))
            _assert_no_value_leak(input_dir, cap_values, capsule=native)
        # Neutralise pointers to the PUBLISHED results (capsule pages, published
        # images, digests, DOIs, hosted result files) for EVERY capsule, then
        # fail loud if any survives — see :mod:`._identifiers` / ADR 0001.
        id_report = mask_hosting_identifiers(input_dir, exact)
        assert_no_hosting_identifiers(input_dir, exact, capsule=native)
        masked_identifiers.extend(id_report["redacted"])
        identifier_unscanned.extend(id_report["unscanned"])
        for name, n in id_report["classes"].items():
            identifier_classes[name] = identifier_classes.get(name, 0) + n

        # task.jsonl — this capsule's rows only, data rewritten to ./input.
        capsule_tasks = [{**task, "data": f"./{_INPUT_SUBDIR}"} for task in rows]
        task_path = capsule_dir / _TASK_FILENAME
        with task_path.open("w", encoding="utf-8", newline="\n") as fh:
            for task in capsule_tasks:
                fh.write(json.dumps(task, sort_keys=True, ensure_ascii=False))
                fh.write("\n")

        # Uniform schema, copied into every capsule dir.
        (capsule_dir / _SCHEMA_FILENAME).write_text(
            json.dumps(submission_schema_for_tasks(rows), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Example pre-filled with this capsule's real task_id(s).
        (capsule_dir / _EXAMPLE_FILENAME).write_text(
            json.dumps(
                [
                    {"task_id": tid, "answer": "<your answer here>"}
                    for tid in row["task_ids"]
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # Plain-language instructions, friendly id in the human-facing text.
        (capsule_dir / _README_FILENAME).write_text(
            _render_capsule_readme(
                friendly_id=row["friendly_id"],
                benchmark=row["benchmark"],
                task_ids=row["task_ids"],
                has_environment=(input_dir / "environment" / "Dockerfile").is_file(),
            ),
            encoding="utf-8",
        )
        # Path-name backstop over the WHOLE capsule dir (input/ + our files).
        assert_no_identifier_paths(capsule_dir, exact, patterns, capsule=native)
        materialized.append(str(capsule_dir))

    renames_path = None
    if eval_dir is not None:
        renames_path = _write_path_renames(Path(eval_dir), path_renames)

    # Publish mapper only after preflight and successful selected writes.
    with index_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in index:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    records = dict(ledger.get("materializations", {}))
    for row in selected:
        name = row["dir"]
        records[name] = {
            **identities[name],
            "materialized_sha256": _materialized_identity(for_solver_dir / name),
        }
    ledger = {
        "contract": _MATERIALIZATION_CONTRACT,
        "source_identity": source_identity,
        "index_sha256": _sha256_file(index_path),
        "materializations": records,
    }
    ledger_path = for_solver_dir / _IDENTITY_FILENAME
    descriptor = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
        os.fchmod(fh.fileno(), 0o600)
        json.dump(
            ledger, fh, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )

    return {
        "index": str(index_path),
        "n_capsules": len(index),
        "n_materialized": len(materialized),
        "n_skipped": len(skipped),
        "materialized": materialized,
        "skipped": skipped,
        "stripped_leaks": stripped_leaks,
        "stripped_files": stripped_files,
        "masked_values": masked_values,
        "cleared_notebooks": cleared_notebooks,
        "masked_identifiers": masked_identifiers,
        "identifier_classes": identifier_classes,
        "identifier_unscanned": identifier_unscanned,
        "path_renames": path_renames,
        "path_renames_file": renames_path,
        "purged_legacy": purged_legacy,
        "only": only,
        "identity_ledger": str(ledger_path),
    }


_PATH_RENAMES_FILENAME = "path_renames.json"


def _write_path_renames(eval_dir: Path, path_renames: dict[str, dict]) -> str:
    """Merge this run's old→new path mapping into ``eval/path_renames.json``.

    Operator-side traceability for :func:`neutralize_identifier_paths`: the
    OLD names carry source identifiers, so they never live under
    ``for_solver/``. Entries of capsules not materialized in this run (cached
    or outside ``only``) are kept.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / _PATH_RENAMES_FILENAME
    merged: dict = {}
    if path.is_file() and not path.is_symlink():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            merged = loaded
    merged.update(path_renames)
    path.write_text(
        json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(path)


def write_eval(
    *,
    eval_dir: Path,
    answers: list[dict],
    evaluate_py_source: str,
) -> dict:
    """Write the operator-side ``eval/`` view.

    Emits ``answers.jsonl`` (one answer per line, ``sort_keys=True``,
    ``ensure_ascii=False``) and ``evaluate.py`` from the supplied source
    string, marking the scorer executable (0o755).
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    answers_path = eval_dir / "answers.jsonl"
    with answers_path.open("w", encoding="utf-8", newline="\n") as fh:
        for answer in answers:
            fh.write(json.dumps(answer, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    evaluate_path = eval_dir / "evaluate.py"
    evaluate_path.write_text(evaluate_py_source, encoding="utf-8")
    evaluate_path.chmod(
        evaluate_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    return {
        "answers": str(answers_path),
        "n_answers": len(answers),
        "evaluate": str(evaluate_path),
    }


__all__ = [
    "UNIFORM_SUBMISSION_SCHEMA",
    "TASK_KEYS",
    "LEAK_DIRS",
    "EVALUATE_PY",
    "render_evaluate_py",
    "write_for_solver",
    "write_for_solver_per_capsule",
    "build_capsule_index",
    "friendly_capsule_id",
    "write_eval",
]

# EOF
