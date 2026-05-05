#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_groups.py

"""Build the ``<domain> <dataset> <action>`` 3-level click group tree.

Every catalog source becomes a noun group with one verb (``fetch``).
HuggingFace adds ``search``, ``info``, and ``download-file`` (so the
noun-group form is justified — see ``general/03_interface_02_cli/02``
on tree-vs-compound-leaf).
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .._sources import (
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
    def _cmd(query, limit, output):
        """Search HuggingFace Hub for datasets.

        \b
        Example:
          $ scitex-dataset general huggingface search "biology"
        """
        from ..general.huggingface import search_hub

        try:
            results = search_hub(query=query, limit=limit)
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


__all__ = ["register_domain_groups"]

# EOF
