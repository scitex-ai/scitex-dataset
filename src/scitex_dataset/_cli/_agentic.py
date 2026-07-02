#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_agentic.py

"""Agentic AI-for-science verb-tree builders for the CLI.

The ``ai-for-science`` benchmarks (corebench / bixbench /
biomysterybench) use a multi-step verb tree rather than a single
``fetch``::

    download | prepare | standardize | validate | score

This module builds those click commands for one benchmark source. It is
kept separate from :mod:`._groups` (the group-tree orchestrator) so each
file owns a single responsibility and stays well under the line budget.

- ``download`` / ``prepare`` / ``standardize`` — the raw → {for_solver,
  eval} preparation pipeline (see :mod:`..ai_for_science`).
- ``validate`` — STRUCTURAL, oracle-free submission check
  (:func:`..ai_for_science._validate.validate_submission`); non-zero
  exit when the submission is invalid.
- ``score`` — CORRECTNESS scoring against the operator oracle
  (:func:`..ai_for_science._score.score_submission`); one record per
  task with a 5-way verdict.
"""

from __future__ import annotations

import json

import click


def _agentic_bench_module(source: str):
    """Return the ``scitex_dataset.ai_for_science.<source>`` module."""
    import importlib

    return importlib.import_module(
        f"..ai_for_science.{source}", package=__package__
    )


def _emit_result(as_json: bool, result: dict, headline: str) -> None:
    """Render a verb result dict — JSON when requested, human-friendly otherwise."""
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    click.echo(headline)
    for key in ("output", "manifest", "raw_dir", "for_solver_dir", "eval_dir"):
        if key in result:
            click.echo(f"  {key}: {result[key]}")
    if "n_tasks" in result:
        click.echo(f"  n_tasks: {result['n_tasks']}")


def _common_paths_options(f):
    """Attach the standard --dataset-root / --json options to a click cmd."""
    f = click.option(
        "--dataset-root",
        type=click.Path(),
        default=None,
        help="Dataset root holding ai-for-science/ (default: "
        "$SCITEX_DATASET_ROOT, nearest project .scitex/dataset, or "
        "~/.scitex/dataset).",
    )(f)
    f = click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit the result dict as JSON.",
    )(f)
    return f


def _make_download_command(source: str, module) -> click.Command:
    @click.command("download")
    @_common_paths_options
    @click.option(
        "--full",
        "download_full",
        is_flag=True,
        help="Pull the full gated set instead of the small preview "
        "(where the benchmark distinguishes them, e.g. biomysterybench). "
        "For corebench, the full set is the oracle-driven 90-capsule "
        "run: it bootstraps the answer manifests (fetch + gpg-decrypt) "
        "then fetches every capsule tarball.",
    )
    @click.option(
        "--capsule-ids",
        default=None,
        help="corebench only: comma-separated NATIVE capsule ids (e.g. "
        "capsule-0201225,capsule-0238624) to fetch just those tarballs "
        "from the CDN — the oracle URLs are never touched. Overrides the "
        "oracle-driven full run.",
    )
    @click.option(
        "--verify-integrity",
        is_flag=True,
        help="Re-check existing files by sha256 before skipping (opt-in; "
        "default skip is by existence, no hashing).",
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Re-download everything, ignoring what's already on disk.",
    )
    def _download_cmd(
        dataset_root, as_json, download_full, capsule_ids, verify_integrity, force
    ):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._base import resolve_paths

        paths = resolve_paths(module.BENCHMARK, dataset_root=dataset_root)
        kwargs = dict(
            raw_dir=paths.raw_dir,
            download_full=download_full,
            verify_integrity=verify_integrity,
            force=force,
        )
        if capsule_ids:
            # corebench's download() takes an explicit capsule_ids list;
            # siblings absorb the extra kwarg via **_ (harmless when unset).
            kwargs["capsule_ids"] = [
                cid.strip() for cid in capsule_ids.split(",") if cid.strip()
            ]
        try:
            result = module.download(**kwargs)
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        result["paths"] = paths.as_dict()
        _emit_result(as_json, result, f"Downloaded {source}.")

    _download_cmd.help = (
        f"Fetch the upstream {source} snapshot into raw/ as-is.\n\n"
        f"WARNING: downloads are multi-GB. Run on SLURM, not a login "
        f"node or CI.\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} download\n"
        f"  $ scitex-dataset ai-for-science {source} download "
        f"--full --dataset-root /scratch/datasets"
    )
    return _download_cmd


def _make_standardize_command(source: str, module) -> click.Command:
    @click.command("standardize")
    @_common_paths_options
    @click.option(
        "--only",
        default=None,
        help="Materialize ONLY this capsule's for_solver/ dir — accepts a "
        "friendly id (capsule-NNN) or a native capsule id (e.g. "
        "capsule-0201225), resolved via the index.jsonl mapper. The full "
        "mapper is still written. Default: all capsules.",
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Re-extract capsules already present in for_solver/ (default "
        "skips by existence).",
    )
    def _standardize_cmd(dataset_root, as_json, only, force):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._base import resolve_paths

        paths = resolve_paths(module.BENCHMARK, dataset_root=dataset_root)
        try:
            result = module.standardize(
                raw_dir=paths.raw_dir,
                for_solver_dir=paths.for_solver_dir,
                eval_dir=paths.eval_dir,
                only=only,
                force=force,
            )
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        result["paths"] = paths.as_dict()
        result["for_solver_dir"] = str(paths.for_solver_dir)
        result["eval_dir"] = str(paths.eval_dir)
        _emit_result(as_json, result, f"Standardized {source}.")

    _standardize_cmd.help = (
        f"Read the {source} raw/ snapshot; write the agent-visible "
        f"for_solver/ view (per-capsule dirs + index.jsonl mapper, no "
        f"answers) + the operator eval/ view (answers.jsonl + "
        f"evaluate.py).\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} standardize\n"
        f"  $ scitex-dataset ai-for-science {source} standardize "
        f"--only capsule-0201225\n"
        f"  $ scitex-dataset ai-for-science {source} standardize --json"
    )
    return _standardize_cmd


def _make_prepare_command(source: str, module) -> click.Command:
    @click.command("prepare")
    @_common_paths_options
    @click.option(
        "--version",
        default="v0-unstamped",
        help="Snapshot version recorded in MANIFEST.yaml (upstream tag / "
        "HF revision / ISO-date).",
    )
    @click.option(
        "--skip-download",
        is_flag=True,
        help="Skip the upstream pull (raw/ already staged).",
    )
    @click.option(
        "--full",
        "download_full",
        is_flag=True,
        help="Pull the full gated set instead of the small preview.",
    )
    @click.option(
        "--verify-integrity",
        is_flag=True,
        help="Re-check existing files by sha256 before skipping (opt-in).",
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Re-download everything, ignoring what's already on disk.",
    )
    def _prepare_cmd(
        dataset_root,
        as_json,
        version,
        skip_download,
        download_full,
        verify_integrity,
        force,
    ):
        """Placeholder docstring (overwritten below with the per-source example)."""
        try:
            result = module.prepare(
                dataset_root=dataset_root,
                version=version,
                skip_download=skip_download,
                download_full=download_full,
                verify_integrity=verify_integrity,
                force=force,
            )
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        _emit_result(as_json, result, f"Prepared {source}.")

    _prepare_cmd.help = (
        f"Prepare {source}: download (raw/) + standardize "
        f"(for_solver/ + eval/) + emit .scitex/dataset/MANIFEST.yaml "
        f"(id + version + sha256 + mask-seed).\n\nWARNING: includes the "
        f"multi-GB download step unless --skip-download is given. "
        f"SLURM-only on shared compute.\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} prepare --skip-download\n"
        f"  $ scitex-dataset ai-for-science {source} prepare --full --version v1.0"
    )
    return _prepare_cmd


def _make_validate_command(source: str, module) -> click.Command:
    @click.command("validate")
    @_common_paths_options
    @click.option(
        "--submission",
        required=True,
        type=click.Path(),
        help="Path to the agent submission JSON (the uniform array).",
    )
    def _validate_cmd(dataset_root, as_json, submission):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._base import resolve_paths
        from ..ai_for_science._validate import (
            expected_task_ids_from_for_solver,
            validate_submission,
        )

        paths = resolve_paths(module.BENCHMARK, dataset_root=dataset_root)
        expected = expected_task_ids_from_for_solver(paths.for_solver_dir)
        result = validate_submission(
            module.BENCHMARK, submission, expected_task_ids=expected
        )
        if as_json:
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            if result["ok"]:
                click.echo(f"OK: {source} submission is structurally valid.")
            else:
                click.echo(f"INVALID: {source} submission has errors:")
            for err in result["errors"]:
                click.echo(f"  [{err['kind']}] {err['path']}: {err['message']}")
        if not result["ok"]:
            raise SystemExit(1)

    _validate_cmd.help = (
        f"Structurally validate a {source} submission (ORACLE-FREE, "
        f"host-side): required fields, task_id shape, and array length "
        f"vs the for_solver index. Non-zero exit if invalid.\n\n\b\n"
        f"Example:\n"
        f"  $ scitex-dataset ai-for-science {source} validate "
        f"--submission sub.json\n"
        f"  $ scitex-dataset ai-for-science {source} validate "
        f"--submission sub.json --json"
    )
    return _validate_cmd


def _make_score_command(source: str, module) -> click.Command:
    @click.command("score")
    @_common_paths_options
    @click.option(
        "--submission",
        required=True,
        type=click.Path(),
        help="Path to the agent submission JSON (the uniform array).",
    )
    def _score_cmd(dataset_root, as_json, submission):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._base import resolve_paths
        from ..ai_for_science._score import score_submission

        paths = resolve_paths(module.BENCHMARK, dataset_root=dataset_root)
        try:
            records = score_submission(
                module.BENCHMARK,
                submission,
                answers=paths.eval_dir,
            )
        except OSError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        if as_json:
            click.echo(json.dumps(records, indent=2, default=str))
        else:
            click.echo(f"Scored {source}: {len(records)} task(s).")
            for rec in records:
                extra = rec.get("malformed_kind", "")
                suffix = f" ({extra})" if extra else ""
                click.echo(f"  {rec['task_id']}: {rec['verdict']}{suffix}")

    _score_cmd.help = (
        f"Score a {source} submission against the operator oracle "
        f"(eval/answers.jsonl; host-side only). Emits one record per "
        f"task with a 5-way verdict (correct/wrong/abstain/malformed/"
        f"needs_rubric).\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} score "
        f"--submission sub.json\n"
        f"  $ scitex-dataset ai-for-science {source} score "
        f"--submission sub.json --json"
    )
    return _score_cmd


def _agentic_bench_commands(source: str) -> list[click.Command]:
    """Build the agentic verb-tree click commands for one benchmark.

    Returns ``[download, prepare, standardize, validate, score]`` — the
    preparation pipeline plus the host-side submission validator and
    scorer.
    """
    module = _agentic_bench_module(source)
    return [
        _make_download_command(source, module),
        _make_prepare_command(source, module),
        _make_standardize_command(source, module),
        _make_validate_command(source, module),
        _make_score_command(source, module),
    ]


__all__ = ["_agentic_bench_commands"]

# EOF
