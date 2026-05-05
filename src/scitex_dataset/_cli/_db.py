#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_db.py

"""``scitex-dataset db ...`` — local SQLite + FTS5 index."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .._sources import CATALOG_SOURCES

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _print_command_help(cmd, prefix: str, parent_ctx) -> None:
    click.echo(f"\n{'=' * 50}")
    click.echo(f"{prefix}")
    click.echo("=" * 50)
    sub_ctx = click.Context(cmd, info_name=prefix.split()[-1], parent=parent_ctx)
    click.echo(cmd.get_help(sub_ctx))
    if isinstance(cmd, click.Group):
        for sub_name, sub_cmd in sorted(cmd.commands.items()):
            _print_command_help(sub_cmd, f"{prefix} {sub_name}", sub_ctx)


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.pass_context
def db(ctx: click.Context, help_recursive: bool):
    """Local database commands for fast offline searching."""
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
    help="Sources to index (default: all catalog sources).",
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
    del yes
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
@click.option("-s", "--source", type=click.Choice(CATALOG_SOURCES))
@click.option("-m", "--modality", help="Filter by modality (mri, eeg, etc.).")
@click.option("--min-subjects", type=int, help="Minimum subjects.")
@click.option("--max-subjects", type=int, help="Maximum subjects.")
@click.option("--min-downloads", type=int, help="Minimum downloads.")
@click.option("-n", "--limit", default=20, help="Max results (default: 20).")
@click.option("--order-by", default="downloads", help="Order by field.")
@click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
@click.option("--json", "as_json", is_flag=True, help="Emit results as JSON to stdout.")
def db_search(
    query,
    source,
    modality,
    min_subjects,
    max_subjects,
    min_downloads,
    limit,
    order_by,
    output,
    as_json,
) -> None:
    """Search the local database.

    \b
    Example:
      $ scitex-dataset db search "EEG"
      $ scitex-dataset db search "MRI" -s openneuro -n 5
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
    """(deprecated) Renamed to ``show-stats``."""
    click.echo(
        "error: `scitex-dataset db stats` was renamed to "
        "`scitex-dataset db show-stats`.",
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
      $ scitex-dataset db clear --yes
    """
    from .. import database

    if database.clear():
        click.echo("Database deleted.")
    else:
        click.echo("Database not found.")


__all__ = ["db"]

# EOF
