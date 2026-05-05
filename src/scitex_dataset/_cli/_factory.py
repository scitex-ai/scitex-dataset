#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_factory.py

"""Factory that turns a source's ``fetch_all_datasets / format_dataset``
pair into a uniform ``fetch`` click command.

Each catalog source under ``scitex_dataset.<domain>.<src>`` exposes the
two-callable contract; the factory wires the standard flags
(``-n / -o / -v`` and, where supported, ``-q``) once instead of
copy-pasting per source.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click

from .._sources import DOMAIN_OF


def _resolve_module(source: str):
    """Locate the ``scitex_dataset.<domain>.<source>`` module."""
    domain = DOMAIN_OF[source]
    return importlib.import_module(f"..{domain}.{source}", package=__package__)


def make_fetch_command(
    source: str,
    *,
    accepts_query: bool = False,
    accepts_batch_size: bool = False,
    label: str = "datasets",
    description: str = "",
) -> click.Command:
    """Build a ``fetch`` click command for ``source``."""
    module = _resolve_module(source)
    fetch_all_datasets = module.fetch_all_datasets
    format_dataset = module.format_dataset
    domain = DOMAIN_OF[source]

    @click.command("fetch")
    @click.option("-n", "--max-datasets", default=0, help="Max datasets (0=all).")
    @click.option("-o", "--output", type=click.Path(), help="Output JSON file.")
    @click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
    @(
        click.option("-q", "--query", default="", help="Search query.")
        if accepts_query
        else (lambda f: f)
    )
    @(
        click.option("-b", "--batch-size", default=100, help="Datasets per request.")
        if accepts_batch_size
        else (lambda f: f)
    )
    def _cmd(max_datasets: int, output, verbose: bool, **extra) -> None:
        kwargs = {"max_datasets": max_datasets if max_datasets > 0 else None}
        if accepts_query:
            kwargs["query"] = extra.get("query", "")
        if accepts_batch_size:
            kwargs["batch_size"] = extra.get("batch_size", 100)

        if verbose:
            click.echo(f"Fetching {label} from {source}...")

        datasets = fetch_all_datasets(**kwargs)

        if not datasets:
            click.echo("No datasets fetched", err=True)
            raise SystemExit(1)

        formatted = [format_dataset(ds) for ds in datasets]

        if output:
            Path(output).write_text(json.dumps(formatted, indent=2))
            click.echo(f"Saved {len(formatted)} {label} to {output}")
        else:
            if verbose:
                for ds in formatted[:10]:
                    name = ds.get("name", "")[:60]
                    click.echo(f"  {ds.get('id', '?')}: {name}")
            click.echo(f"Fetched {len(formatted)} {label}")

    example = (
        f"\n\n\\b\nExample:\n"
        f"  $ scitex-dataset {domain} {source} fetch\n"
        f"  $ scitex-dataset {domain} {source} fetch -n 10 -o {source}.json"
    )
    base_help = description or f"Fetch {label} from {source}."
    _cmd.help = base_help + example
    return _cmd


__all__ = ["make_fetch_command"]

# EOF
