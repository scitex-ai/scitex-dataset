#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-30 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/_cli/__init__.py

"""Command-line interface for scitex-dataset."""

import json
from pathlib import Path

import click

from .. import __version__
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
@click.option("--version", "-V", is_flag=True, help="Show version and exit.")
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.pass_context
def main(ctx: click.Context, version: bool, help_recursive: bool) -> None:
    """scitex-dataset - Unified interface for scientific dataset discovery.

    Fetch and search datasets from neuroscience repositories:
    OpenNeuro, DANDI, PhysioNet, and more.
    """
    if version:
        click.echo(f"scitex-dataset {__version__}")
        ctx.exit(0)

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


# OpenNeuro command
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-b", "--batch-size", default=100, help="Datasets per request.")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def openneuro(max_datasets: int, batch_size: int, output: str, verbose: bool) -> None:
    """Fetch datasets from OpenNeuro (BIDS neuroimaging)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def dandi(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from DANDI Archive (NWB neurophysiology)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def physionet(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from PhysioNet (EEG/ECG/physiology)."""
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
@main.command()
@click.option("-q", "--query", default="", help="Search query.")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def zenodo(query: str, max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from Zenodo (general scientific repository)."""
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
@main.command()
@click.option("-q", "--query", default="", help="Search query.")
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def figshare(query: str, max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from Figshare (research data sharing)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def openml(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from OpenML (machine learning datasets)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def moleculenet(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from MoleculeNet (molecular ML benchmarks)."""
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
                click.echo(f"  {ds['id']}: {ds['name']} ({ds.get('n_compounds', '?')} compounds)")
        click.echo(f"Fetched {len(formatted)} datasets")


# GEO command
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def geo(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch datasets from GEO (Gene Expression Omnibus)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def chembl(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch assays from ChEMBL (bioactivity database)."""
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
@main.command()
@click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def clinicaltrials(max_datasets: int, output: str, verbose: bool) -> None:
    """Fetch studies from ClinicalTrials.gov."""
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
    type=click.Choice(
        ["openneuro", "dandi", "physionet", "zenodo", "figshare", "openml", "geo", "chembl", "moleculenet", "clinicaltrials"]
    ),
    help="Sources to index (default: all).",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def db_build(sources: tuple, verbose: bool) -> None:
    """Build/rebuild the local dataset database."""
    from .. import database

    source_list = list(sources) if sources else None

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
    type=click.Choice(
        ["openneuro", "dandi", "physionet", "zenodo", "figshare", "openml", "geo", "chembl", "moleculenet", "clinicaltrials"]
    ),
)
@click.option("-m", "--modality", help="Filter by modality (mri, eeg, etc.).")
@click.option("--min-subjects", type=int, help="Minimum subjects.")
@click.option("--max-subjects", type=int, help="Maximum subjects.")
@click.option("--min-downloads", type=int, help="Minimum downloads.")
@click.option("-n", "--limit", default=20, help="Max results (default: 20).")
@click.option("--order-by", default="downloads", help="Order by field.")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
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
) -> None:
    """Search the local database."""
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


@db.command("stats")
def db_stats() -> None:
    """Show database statistics."""
    from .. import database

    stats = database.get_stats()

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
    """Delete the local database."""
    from .. import database

    if database.clear():
        click.echo("Database deleted.")
    else:
        click.echo("Database not found.")


# Shell completion
@main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Generate shell completion script.

    \b
    Usage:
      eval "$(scitex-dataset completion bash)"   # Bash
      eval "$(scitex-dataset completion zsh)"    # Zsh
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
