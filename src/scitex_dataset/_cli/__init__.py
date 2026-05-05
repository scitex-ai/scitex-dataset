#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-30 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/_cli/__init__.py

"""Command-line interface for scitex-dataset."""

import json
from pathlib import Path

import click

from .. import __version__
from .._sources import CATALOG_SOURCES
from ._introspect import list_python_apis
from ._mcp_commands import mcp

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _print_command_help(cmd, prefix: str, parent_ctx) -> None:
    """Recursively print help for a command and its subcommands."""
    click.echo(f"\n{'=' * 50}")
    click.echo(f"{prefix}")
    click.echo("=" * 50)
    sub_ctx = click.Context(cmd, info_name=prefix.split()[-1], parent=parent_ctx)
    click.echo(cmd.get_help(sub_ctx))

    if isinstance(cmd, click.Group):
        for sub_name, sub_cmd in sorted(cmd.commands.items()):
            _print_command_help(sub_cmd, f"{prefix} {sub_name}", sub_ctx)


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="scitex-dataset",
    message="%(prog)s %(version)s",
)
@click.help_option("-h", "--help")
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit structured JSON output (propagates to subcommands that honour it).",
)
@click.pass_context
def main(ctx: click.Context, help_recursive: bool, as_json: bool) -> None:
    """scitex-dataset - Unified interface for scientific dataset discovery.

    Fetch and search datasets from neuroscience repositories:
    OpenNeuro, DANDI, PhysioNet, and more.

    \b
    Config is loaded with the SciTeX precedence chain:
      config.yaml -> $SCITEX_DATASET_CONFIG -> ~/.scitex/dataset/config.yaml -> defaults
    """
    ctx.ensure_object(dict)
    ctx.obj["as_json"] = as_json

    if help_recursive:
        click.echo(f"scitex-dataset {__version__}")
        click.echo(main.get_help(ctx))
        for name, cmd in sorted(main.commands.items()):
            _print_command_help(cmd, f"scitex-dataset {name}", ctx)
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register subcommand groups
main.add_command(mcp)
main.add_command(list_python_apis)

try:
    from scitex_dev.cli import docs_click_group

    main.add_command(docs_click_group(package="scitex-dataset"))
except ImportError:
    pass


# Hidden deprecation redirects: bare `<source>` → real subcommand.
# Most sources map to compound leaves (`fetch-<source>`); HuggingFace
# uses a noun-group (`hf <verb>`) because it has 4 verbs.
_DEPRECATED_TARGETS = {src: f"fetch-{src}" for src in CATALOG_SOURCES}
_DEPRECATED_TARGETS["huggingface"] = "hf"


def _dataset_deprecated(name: str, target: str):
    """Hidden top-level redirect: ``<name>`` → ``<target>``."""

    @click.pass_context
    def _impl(ctx, **_):
        click.echo(
            f"error: `scitex-dataset {name}` was renamed to "
            f"`scitex-dataset {target}`.\n"
            f"Re-run with: scitex-dataset {target} [...]",
            err=True,
        )
        ctx.exit(2)

    return click.command(
        name,
        hidden=True,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )(_impl)


for _src, _tgt in _DEPRECATED_TARGETS.items():
    main.add_command(_dataset_deprecated(_src, _tgt))


# OpenNeuro command
@main.command("fetch-openneuro")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-b", "--batch-size", default=100, help="Datasets per request.")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def openneuro(max_datasets: int, batch_size: int, output: str, verbose: bool) -> None:
    """Fetch datasets from OpenNeuro (BIDS neuroimaging).

    \b
    Example:
      $ scitex-dataset fetch-openneuro
      $ scitex-dataset fetch-openneuro -n 10 -o openneuro.json
    """
    from ..neuroscience.openneuro import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching datasets from OpenNeuro...")

    datasets = fetch_all_datasets(
        batch_size=batch_size,
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name']} ({ds['n_subjects']} subjects)")
        click.echo(f"Fetched {len(formatted)} datasets")


# DANDI command
@main.command("fetch-dandi")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def dandi(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from DANDI Archive (NWB neurophysiology).

    \b
    Example:
      $ scitex-dataset fetch-dandi
      $ scitex-dataset fetch-dandi -n 10 -o dandi.json
    """
    from ..neuroscience.dandi import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching dandisets from DANDI Archive...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} dandisets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name']}")
        click.echo(f"Fetched {len(formatted)} dandisets")


# PhysioNet command
@main.command("fetch-physionet")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def physionet(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from PhysioNet (EEG/ECG/physiology).

    \b
    Example:
      $ scitex-dataset fetch-physionet
      $ scitex-dataset fetch-physionet -n 10 -o physionet.json
    """
    from ..neuroscience.physionet import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching databases from PhysioNet...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} databases to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name']}")
        click.echo(f"Fetched {len(formatted)} databases")


# Zenodo command
@main.command("fetch-zenodo")
@click.option("-q", "--query", default="", help="Search query.")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def zenodo(query: str, max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from Zenodo (general scientific repository).

    \b
    Example:
      $ scitex-dataset fetch-zenodo -q "neural network"
      $ scitex-dataset fetch-zenodo -n 10 -o zenodo.json
    """
    from ..general.zenodo import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching datasets from Zenodo...")
        if query:
            click.echo(f"  Query: {query}")

    datasets = fetch_all_datasets(
        query=query,
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} datasets")


# Figshare command
@main.command("fetch-figshare")
@click.option("-q", "--query", default="", help="Search query.")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def figshare(query: str, max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from Figshare (research data sharing).

    \b
    Example:
      $ scitex-dataset fetch-figshare -q "biology"
      $ scitex-dataset fetch-figshare -n 10 -o figshare.json
    """
    from ..general.figshare import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching datasets from Figshare...")

    datasets = fetch_all_datasets(
        query=query,
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} datasets")


# OpenML command
@main.command("fetch-openml")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def openml(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from OpenML (machine learning datasets).

    \b
    Example:
      $ scitex-dataset fetch-openml
      $ scitex-dataset fetch-openml -n 10 -o openml.json
    """
    from ..general.openml import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching datasets from OpenML...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} datasets")


# MoleculeNet command
@main.command("fetch-moleculenet")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def moleculenet(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from MoleculeNet (molecular ML benchmarks).

    \b
    Example:
      $ scitex-dataset fetch-moleculenet
      $ scitex-dataset fetch-moleculenet -n 10 -o moleculenet.json
    """
    from ..pharmacology.moleculenet import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching MoleculeNet benchmark catalog...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(
                    f"  {ds['id']}: {ds['name']} ({ds.get('n_compounds', '?')} compounds)"
                )
        click.echo(f"Fetched {len(formatted)} datasets")


# GEO command
@main.command("fetch-geo")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def geo(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from GEO (Gene Expression Omnibus).

    \b
    Example:
      $ scitex-dataset fetch-geo
      $ scitex-dataset fetch-geo -n 10 -o geo.json
    """
    from ..biology.geo import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching datasets from GEO...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} datasets to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} datasets")


# ChEMBL command
@main.command("fetch-chembl")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def chembl(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch assays from ChEMBL (bioactivity database).

    \b
    Example:
      $ scitex-dataset fetch-chembl
      $ scitex-dataset fetch-chembl -n 10 -o chembl.json
    """
    from ..pharmacology.chembl import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching assays from ChEMBL...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} assays to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} assays")


# ClinicalTrials command
@main.command("fetch-clinicaltrials")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def clinicaltrials(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch studies from ClinicalTrials.gov.

    \b
    Example:
      $ scitex-dataset fetch-clinicaltrials
      $ scitex-dataset fetch-clinicaltrials -n 10 -o clinicaltrials.json
    """
    from ..medical.clinicaltrials import fetch_all_datasets, format_dataset

    if verbose:
        click.echo("Fetching studies from ClinicalTrials.gov...")

    datasets = fetch_all_datasets(
        max_datasets=max_datasets if max_datasets > 0 else None,
    )

    if not datasets:
        click.echo("No datasets fetched", err=True)
        raise SystemExit(1)

    formatted = [format_dataset(ds) for ds in datasets]

    if output:
        Path(output).write_text(json.dumps(formatted, indent=2))
        click.echo(f"Saved {len(formatted)} studies to {output}")
    else:
        if verbose:
            for ds in formatted[:10]:
                click.echo(f"  {ds['id']}: {ds['name'][:50]}")
        click.echo(f"Fetched {len(formatted)} studies")


# Database commands
@main.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.pass_context
def db(ctx: click.Context, help_recursive: bool):
    """Local database commands for fast searching."""
    if help_recursive:
        click.echo(db.get_help(ctx))
        for name, cmd in sorted(db.commands.items()):
            _print_command_help(cmd, f"scitex-dataset db {name}", ctx)
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@db.command("build")
@click.option(
    "-s",
    "--sources",
    multiple=True,
    type=click.Choice(CATALOG_SOURCES),
    help="Sources to index (default: all).",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Print build plan without writing.")
@click.option(
    "-y", "--yes", is_flag=True, help="Suppress interactive confirmation (assume yes)."
)
def db_build(sources: tuple, verbose: bool, dry_run: bool, yes: bool) -> None:
    """Build/rebuild the local dataset database.

    \b
    Example:
      $ scitex-dataset db build
      $ scitex-dataset db build -s openneuro -s dandi
      $ scitex-dataset db build --dry-run
    """
    from .. import database

    source_list = list(sources) if sources else None

    if dry_run:
        click.echo(
            f"DRY RUN — would build database at {database.get_db_path()} "
            f"(sources={source_list or 'all'})"
        )
        return

    if verbose:
        click.echo(f"Building database at {database.get_db_path()}")
        click.echo(f"Sources: {source_list or 'all'}")

    counts = database.build(sources=source_list)

    click.echo("Database built:")
    for src, count in counts.items():
        click.echo(f"  {src}: {count} datasets")
    click.echo(f"Total: {sum(counts.values())} datasets")


@db.command("search")
@click.argument("query", required=False)
@click.option(
    "-s",
    "--source",
    type=click.Choice(CATALOG_SOURCES),
)
@click.option("-m", "--modality", help="Filter by modality (mri, eeg, etc.).")
@click.option("--min-subjects", type=int, help="Minimum subjects.")
@click.option("--max-subjects", type=int, help="Maximum subjects.")
@click.option("--min-downloads", type=int, help="Minimum downloads.")
@click.option("-n", "--limit", default=20, help="Max results (default: 20).")
@click.option("--order-by", default="downloads", help="Order by field.")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("--json", "as_json", is_flag=True, help="Emit results as JSON to stdout.")
def db_search(
    query: str,
    source: str,
    modality: str,
    min_subjects: int,
    max_subjects: int,
    min_downloads: int,
    limit: int,
    order_by: str,
    output: str,
    as_json: bool,
) -> None:
    """Search the local database.

    \b
    Example:
      $ scitex-dataset db search "EEG"
      $ scitex-dataset db search "MRI" -s openneuro -n 5
      $ scitex-dataset db search "EEG" --json
    """
    from .. import database

    results = database.search(
        query=query,
        source=source,
        modality=modality,
        min_subjects=min_subjects,
        max_subjects=max_subjects,
        min_downloads=min_downloads,
        limit=limit,
        order_by=order_by,
    )

    if as_json:
        click.echo(json.dumps({"total": len(results), "results": results}, indent=2))
        return

    if not results:
        click.echo("No datasets found.")
        return

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"Saved {len(results)} results to {output}")
    else:
        for ds in results:
            n_sub = ds.get("n_subjects", 0)
            downloads = ds.get("downloads", 0)
            click.echo(f"  {ds['id']}: {ds.get('name', 'N/A')[:50]}")
            click.echo(f"    subjects={n_sub}, downloads={downloads}")
        click.echo(f"\nFound {len(results)} datasets")


@db.command("stats", hidden=True, context_settings={"ignore_unknown_options": True})
@click.pass_context
def db_stats_deprecated(ctx):
    """(deprecated) Renamed to `show-stats`."""
    click.echo(
        "error: `scitex-dataset db stats` was renamed to "
        "`scitex-dataset db show-stats`.\n"
        "Re-run with: scitex-dataset db show-stats",
        err=True,
    )
    ctx.exit(2)


@db.command("show-stats")
@click.option("--json", "as_json", is_flag=True, help="Emit stats as JSON to stdout.")
def db_show_stats(as_json: bool) -> None:
    """Show database statistics.

    \b
    Example:
      $ scitex-dataset db show-stats
      $ scitex-dataset db show-stats --json
    """
    from .. import database

    stats = database.get_stats()

    if as_json:
        click.echo(json.dumps(stats, indent=2, default=str))
        return

    if not stats.get("exists"):
        click.echo(stats.get("message", "Database not found."))
        click.echo("Run: scitex-dataset db build")
        return

    click.echo(f"Database: {stats['path']}")
    click.echo(f"Size: {stats['size_mb']} MB")
    click.echo(f"Total datasets: {stats['total_datasets']}")
    click.echo(f"Last build: {stats.get('last_build', 'N/A')}")
    click.echo("\nBy source:")
    for src, count in stats.get("by_source", {}).items():
        click.echo(f"  {src}: {count}")


@db.command("clear")
@click.confirmation_option(prompt="Delete the local database?")
def db_clear() -> None:
    """Delete the local database.

    \b
    Example:
      $ scitex-dataset db clear
    """
    from .. import database

    if database.clear():
        click.echo("Database deleted.")
    else:
        click.echo("Database not found.")


# HuggingFace commands
@main.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.pass_context
def hf(ctx: click.Context, help_recursive: bool):
    """HuggingFace dataset and model commands."""
    if help_recursive:
        click.echo(hf.get_help(ctx))
        for name, cmd in sorted(hf.commands.items()):
            _print_command_help(cmd, f"scitex-dataset hf {name}", ctx)
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@hf.command("fetch")
@click.argument("repo_id")
@click.option(
    "-d", "--local-dir", type=click.Path(), help="Local directory for download."
)
@click.option(
    "-t",
    "--repo-type",
    default="dataset",
    type=click.Choice(["dataset", "model"]),
    help="Repository type (default: dataset).",
)
@click.option(
    "-w", "--max-workers", default=4, type=int, help="Parallel workers (default: 4)."
)
@click.option("--hf-home", type=click.Path(), help="Override HF_HOME cache directory.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def hf_fetch(
    repo_id: str,
    local_dir: str,
    repo_type: str,
    max_workers: int,
    hf_home: str,
    verbose: bool,
) -> None:
    """Fetch a HuggingFace dataset or model to disk.

    \b
    Example:
      $ scitex-dataset hf fetch Anthropic/BioMysteryBench-full
      $ scitex-dataset hf fetch Anthropic/BioMysteryBench-full -d /data/gpfs/projects/punim2354/cohort_c_biomysterybench/_full
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


@hf.command("search")
@click.argument("query")
@click.option("-n", "--limit", default=50, type=int, help="Max results (default: 50).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
def hf_search(query: str, limit: int, output: str) -> None:
    """Search for datasets on HuggingFace.

    \b
    Example:
      $ scitex-dataset hf search "biology"
      $ scitex-dataset hf search "neuroimaging" -n 10
    """
    from ..general.huggingface import search_datasets

    try:
        results = search_datasets(query=query, limit=limit)

        if output:
            Path(output).write_text(json.dumps(results, indent=2))
            click.echo(f"Saved {len(results)} results to {output}")
        else:
            for ds in results:
                click.echo(f"  {ds['id']}: {ds['name']}")
                if ds.get("description"):
                    click.echo(f"    {ds['description'][:100]}...")
            click.echo(f"\nFound {len(results)} datasets")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@hf.command("info")
@click.argument("repo_id")
@click.option(
    "-t",
    "--repo-type",
    default="dataset",
    type=click.Choice(["dataset", "model"]),
    help="Repository type (default: dataset).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def hf_info(repo_id: str, repo_type: str, as_json: bool) -> None:
    """Get metadata about a HuggingFace dataset or model.

    \b
    Example:
      $ scitex-dataset hf info Anthropic/BioMysteryBench-full
      $ scitex-dataset hf info username/dataset_name --json
    """
    from ..general.huggingface import dataset_info

    try:
        info = dataset_info(repo_id=repo_id, repo_type=repo_type)

        if as_json:
            click.echo(json.dumps(info, indent=2, default=str))
        else:
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


@hf.command("download-file")
@click.argument("repo_id")
@click.argument("filename")
@click.option(
    "-d", "--local-dir", type=click.Path(), help="Local directory for download."
)
@click.option(
    "-t",
    "--repo-type",
    default="dataset",
    type=click.Choice(["dataset", "model"]),
    help="Repository type (default: dataset).",
)
def hf_download_file(
    repo_id: str, filename: str, local_dir: str, repo_type: str
) -> None:
    """Download a single file from HuggingFace.

    \b
    Example:
      $ scitex-dataset hf download-file Anthropic/BioMysteryBench-full README.md
      $ scitex-dataset hf download-file username/dataset_name data/train.csv -d /local/path
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


# Shell tab-completion
@main.command(
    "completion",
    hidden=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def completion_deprecated(ctx):
    """(deprecated) Renamed to `print-tab-completion`."""
    click.echo(
        "error: `scitex-dataset completion` was renamed to "
        "`scitex-dataset print-tab-completion`.\n"
        "Re-run with: scitex-dataset print-tab-completion --shell bash|zsh|fish",
        err=True,
    )
    ctx.exit(2)


@main.command("print-tab-completion")
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default="bash",
    help="Target shell. Default: bash.",
)
def print_tab_completion(shell: str) -> None:
    """Print tab-completion script to stdout.

    \b
    Example:
      $ eval "$(scitex-dataset print-tab-completion --shell bash)"
      $ scitex-dataset print-tab-completion --shell zsh
    """
    import os
    import subprocess
    import sys

    env = {"_SCITEX_DATASET_COMPLETE": f"{shell}_source"}
    result = subprocess.run(
        [sys.executable, "-m", "scitex_dataset._cli"],
        env={**dict(os.environ), **env},
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout)


if __name__ == "__main__":
    main()

# EOF
