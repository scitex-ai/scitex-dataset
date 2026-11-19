#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_reference_verbs.py

"""``general reference (list|fetch)`` — well-known reference files.

The registry lives in :mod:`scitex_dataset.general.reference`; ``fetch``
verifies the pinned sha256, reuses a verified cached copy, and prints the
provenance record (``--json`` for the full machine-readable dict).
"""

from __future__ import annotations

import json

import click


def _list_command() -> click.Command:
    @click.command("list")
    @click.option("--json", "as_json", is_flag=True, help="Emit entries as JSON.")
    def _cmd(as_json):
        """List the registered reference files.

        \b
        Example:
          $ scitex-dataset general reference list
          $ scitex-dataset general reference list --json
        """
        from ..general.reference import list_references

        entries = list_references()
        if as_json:
            click.echo(json.dumps(entries, indent=2))
            return
        for e in entries:
            click.echo(f"  {e['id']:<12} {e['name']} ({e['size_bytes']:,} bytes)")
            click.echo(f"  {'':<12} {e['url']}")
        click.echo(f"\n{len(entries)} reference file(s)")

    return _cmd


def _fetch_command() -> click.Command:
    @click.command("fetch")
    @click.argument("name")
    @click.option(
        "-d",
        "--dest",
        required=True,
        type=click.Path(file_okay=False),
        help="Directory to keep the file in (created if missing).",
    )
    @click.option(
        "--replace",
        is_flag=True,
        help="Replace an existing file whose sha256 does not match (default: error).",
    )
    @click.option(
        "--timeout", default=60.0, type=float, help="HTTP timeout in seconds."
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit the record as JSON.")
    def _cmd(name, dest, replace, timeout, as_json):
        """Fetch a reference file (sha256-verified, cached) and print its provenance.

        \b
        Example:
          $ scitex-dataset general reference fetch mnist --dest ./data
          $ scitex-dataset general reference fetch mnist --dest ./data --json
        """
        from ..general.reference import fetch_reference

        try:
            rec = fetch_reference(name, dest, timeout=timeout, replace=replace)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)
        if as_json:
            click.echo(json.dumps(rec.to_dict(), indent=2))
            return
        state = "cached (verified)" if rec.cached else "downloaded (verified)"
        click.echo(f"{state}: {rec.path}")
        click.echo(f"  sha256: {rec.sha256}")
        click.echo(f"  size:   {rec.size_bytes:,} bytes")
        click.echo(f"  url:    {rec.url}")
        if rec.license:
            click.echo(f"  license:  {rec.license}")
        if rec.citation:
            click.echo(f"  citation: {rec.citation}")

    return _cmd


def reference_commands() -> list:
    """The verbs of the ``general reference`` noun group."""
    return [_list_command(), _fetch_command()]


__all__ = ["reference_commands"]

# EOF
