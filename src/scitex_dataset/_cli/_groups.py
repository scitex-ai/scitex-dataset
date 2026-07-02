#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_groups.py

"""Build the ``<domain> <dataset> <action>`` 3-level click group tree.

Every catalog source becomes a noun group with one verb (``fetch``).
HuggingFace adds ``search``, ``info``, and ``download-file`` (so the
noun-group form is justified — see ``general/03_interface_02_cli/02``
on tree-vs-compound-leaf).

Agentic-bench sources (``ai-for-science``: corebench / bixbench /
biomysterybench) use a different verb tree (``download | prepare |
standardize``) because the agentic pipeline is multi-step rather than a
single fetch. Their groups are built by ``_make_agentic_bench_group``;
the catalog/ondemand path goes through ``make_fetch_command``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .._sources import (
    AGENTIC_BENCH_SOURCES,
    DOMAINS,
    SOURCE_INFO,
    sources_in_domain,
)
from ._factory import make_fetch_command

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Per-source flag specs (shape of ``fetch_all_datasets``).
_FETCH_SPECS = {
    "openneuro": dict(accepts_batch_size=True, label="datasets"),
    "dandi": dict(label="dandisets"),
    "physionet": dict(label="databases"),
    "zenodo": dict(accepts_query=True, label="datasets"),
    "figshare": dict(accepts_query=True, label="datasets"),
    "openml": dict(label="datasets"),
    "moleculenet": dict(label="datasets"),
    "geo": dict(label="datasets"),
    "chembl": dict(label="assays"),
    "clinicaltrials": dict(label="studies"),
    "huggingface": dict(accepts_query=True, label="datasets"),
}


def _make_dataset_group(source: str) -> click.Group:
    """One noun group per dataset source — holds one or more verbs."""

    info = SOURCE_INFO[source]

    @click.group(
        source,
        invoke_without_command=True,
        context_settings=CONTEXT_SETTINGS,
        help=f"{info['name']} — {info['description']}.",
    )
    @click.pass_context
    def _grp(ctx):
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    spec = _FETCH_SPECS.get(source, {})
    if source == "huggingface":
        # HF fetch takes a positional repo_id; build it bespoke.
        _grp.add_command(_hf_fetch_command())
        _grp.add_command(_hf_search_command())
        _grp.add_command(_hf_info_command())
        _grp.add_command(_hf_download_file_command())
    elif source in AGENTIC_BENCH_SOURCES:
        # Agentic benchmarks use download | prepare | standardize, not fetch.
        for cmd in _agentic_bench_commands(source):
            _grp.add_command(cmd)
    else:
        _grp.add_command(make_fetch_command(source, **spec))

    return _grp


def _make_domain_group(domain: str) -> click.Group:
    """One group per domain (neuroscience, general, biology, ...)."""

    members = sources_in_domain(domain)
    summary = ", ".join(SOURCE_INFO[s]["name"] for s in members)

    @click.group(
        domain,
        invoke_without_command=True,
        context_settings=CONTEXT_SETTINGS,
        help=f"{domain.capitalize()} datasets — {summary}.",
    )
    @click.pass_context
    def _grp(ctx):
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    for src in members:
        _grp.add_command(_make_dataset_group(src))
    return _grp


def register_domain_groups(main: click.Group) -> None:
    """Attach all 5 domain groups to the top-level ``main`` group."""
    for domain in DOMAINS:
        main.add_command(_make_domain_group(domain))


# ---------------------------------------------------------------------------
# HuggingFace verbs — kept here because they don't fit the catalog factory.
# ---------------------------------------------------------------------------


def _hf_fetch_command() -> click.Command:
    @click.command("fetch")
    @click.argument("repo_id")
    @click.option("-d", "--local-dir", type=click.Path(), help="Local directory.")
    @click.option(
        "-t",
        "--repo-type",
        default="dataset",
        type=click.Choice(["dataset", "model"]),
        help="Repository type.",
    )
    @click.option("-w", "--max-workers", default=4, type=int, help="Parallel workers.")
    @click.option("--hf-home", type=click.Path(), help="Override HF_HOME cache dir.")
    @click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
    def _cmd(repo_id, local_dir, repo_type, max_workers, hf_home, verbose):
        """Snapshot-download an HF dataset or model.

        \b
        Example:
          $ scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
        """
        from ..general.huggingface import fetch_dataset

        if verbose:
            click.echo(f"Fetching {repo_type} {repo_id}...")
        try:
            result_path = fetch_dataset(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type=repo_type,
                max_workers=max_workers,
                hf_home_override=hf_home,
            )
            click.echo(f"Downloaded to: {result_path}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    return _cmd


def _hf_search_command() -> click.Command:
    @click.command("search")
    @click.argument("query")
    @click.option("-n", "--limit", default=50, type=int, help="Max results.")
    @click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
    @click.option("--json", "as_json", is_flag=True, help="Emit results as JSON.")
    def _cmd(query, limit, output, as_json):
        """Search HuggingFace Hub for datasets.

        \b
        Example:
          $ scitex-dataset general huggingface search "biology"
          $ scitex-dataset general huggingface search "EEG" --json
        """
        from ..general.huggingface import search_hub

        try:
            results = search_hub(query=query, limit=limit)
            if as_json:
                click.echo(
                    json.dumps({"total": len(results), "results": results}, indent=2)
                )
                return
            if output:
                Path(output).write_text(json.dumps(results, indent=2))
                click.echo(f"Saved {len(results)} results to {output}")
            else:
                for ds in results:
                    click.echo(f"  {ds['id']}: {ds.get('name', '')}")
                click.echo(f"\nFound {len(results)} datasets")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    return _cmd


def _hf_info_command() -> click.Command:
    @click.command("info")
    @click.argument("repo_id")
    @click.option(
        "-t",
        "--repo-type",
        default="dataset",
        type=click.Choice(["dataset", "model"]),
        help="Repository type.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    def _cmd(repo_id, repo_type, as_json):
        """Inspect HF dataset/model metadata.

        \b
        Example:
          $ scitex-dataset general huggingface info Anthropic/BioMysteryBench-full
        """
        from ..general.huggingface import dataset_info

        try:
            info = dataset_info(repo_id=repo_id, repo_type=repo_type)
            if as_json:
                click.echo(json.dumps(info, indent=2, default=str))
                return
            click.echo(f"ID: {info['id']}")
            click.echo(f"Name: {info['name']}")
            if info.get("description"):
                click.echo(f"Description: {info['description']}")
            click.echo(f"Downloads: {info.get('downloads', 0)}")
            click.echo(f"Likes: {info.get('likes', 0)}")
            click.echo(f"Private: {info.get('private', False)}")
            click.echo(f"Gated: {info.get('gated', False)}")
            click.echo(f"URL: {info['url']}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    return _cmd


def _hf_download_file_command() -> click.Command:
    @click.command("download-file")
    @click.argument("repo_id")
    @click.argument("filename")
    @click.option("-d", "--local-dir", type=click.Path(), help="Local directory.")
    @click.option(
        "-t",
        "--repo-type",
        default="dataset",
        type=click.Choice(["dataset", "model"]),
        help="Repository type.",
    )
    def _cmd(repo_id, filename, local_dir, repo_type):
        """Download a single file from an HF repo.

        \b
        Example:
          $ scitex-dataset general huggingface download-file Anthropic/BioMysteryBench-full README.md
        """
        from ..general.huggingface import download_file

        try:
            file_path = download_file(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                repo_type=repo_type,
            )
            click.echo(f"Downloaded to: {file_path}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    return _cmd


# ---------------------------------------------------------------------------
# Agentic AI-for-science benchmarks — download | prepare | standardize verb tree.
# ---------------------------------------------------------------------------


def _agentic_bench_module(source: str):
    """Return the ``scitex_dataset.ai_for_science.<source>`` module."""
    import importlib

    return importlib.import_module(f"..ai_for_science.{source}", package=__package__)


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


def _agentic_bench_commands(source: str) -> list[click.Command]:
    """Build [download, prepare, standardize] click commands for one benchmark."""
    module = _agentic_bench_module(source)

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

    return [_download_cmd, _prepare_cmd, _standardize_cmd]


__all__ = ["register_domain_groups"]

# EOF
