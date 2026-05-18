#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/__init__.py

"""Command-line interface for ``scitex-dataset``.

The command grammar is::

    scitex-dataset <domain> <dataset> <action> [OPTIONS]

For example::

    scitex-dataset neuroscience openneuro fetch -n 50
    scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
    scitex-dataset pharmacology chembl fetch
    scitex-dataset db build

The flat ``fetch-<source>`` and ``hf <verb>`` shapes from earlier
versions are kept as hidden deprecation aliases that print the new path
and exit with status 2.

See ``general/03_interface_02_cli/02_subcommand-structure-noun-verb.md``
for the SciTeX CLI grammar.
"""

from __future__ import annotations

import click

from .. import __version__
from .._sources import ALL_SOURCES, DOMAIN_OF, DOMAINS, SOURCE_INFO
from ._db import db as db_group
from ._groups import register_domain_groups
from ._introspect import list_python_apis
from ._mcp_commands import mcp
from ._skills import skills as skills_group

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


def _repositories_block() -> str:
    """Render the per-domain bullet list shown in top-level ``--help``."""
    lines = ["Available repositories:"]
    for domain in DOMAINS:
        members = [s for s, d in DOMAIN_OF.items() if d == domain]
        if not members:
            continue
        lines.append(f"  {domain}:")
        for src in members:
            info = SOURCE_INFO[src]
            lines.append(f"    - {src:<16} {info['description']}")
    lines.append("")
    lines.append("Run any of:")
    lines.append("  scitex-dataset <domain> <dataset> fetch [OPTIONS]")
    lines.append("  scitex-dataset general huggingface (search|info|download-file)")
    lines.append("  scitex-dataset db (build|search|show-stats|clear)")
    return "\n".join(lines)


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
    """scitex-dataset — unified scientific dataset discovery.

    \b
    Command grammar:
      scitex-dataset <domain> <dataset> <action> [OPTIONS]

    \b
    Config resolution (highest first):
      --config <path>
      $SCITEX_DATASET_CONFIG
      <project>/.scitex/dataset/config.yaml
      $SCITEX_DIR/dataset/config.yaml   (default ~/.scitex/dataset/config.yaml)
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
        click.echo()
        click.echo(_repositories_block())


# ---------------------------------------------------------------------------
# Register subcommands
# ---------------------------------------------------------------------------

main.add_command(mcp)
main.add_command(list_python_apis)
main.add_command(db_group)
main.add_command(skills_group)
register_domain_groups(main)

try:
    from scitex_dev.cli import docs_click_group

    main.add_command(docs_click_group(package="scitex-dataset"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Hidden deprecation aliases
# ---------------------------------------------------------------------------


# Old: bare `<source>` → new: `<domain> <source> fetch` (or `general
# huggingface <verb>` for HF). Hidden so they don't clutter --help.
def _legacy_redirect(name: str, target: str) -> click.Command:
    @click.command(
        name,
        hidden=True,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.pass_context
    def _impl(ctx, **_):
        click.echo(
            f"error: `scitex-dataset {name}` was renamed to "
            f"`scitex-dataset {target}`.\n"
            f"Re-run with: scitex-dataset {target} [...]",
            err=True,
        )
        ctx.exit(2)

    return _impl


for _src in ALL_SOURCES:
    domain = DOMAIN_OF[_src]
    target = f"{domain} {_src} fetch"
    main.add_command(_legacy_redirect(_src, target))
    main.add_command(_legacy_redirect(f"fetch-{_src}", target))

# Old `hf <verb>` form
for _verb in ("fetch", "search", "info", "download-file"):
    main.add_command(_legacy_redirect(f"hf-{_verb}", f"general huggingface {_verb}"))


@main.command(
    "hf",
    hidden=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def _hf_legacy(ctx):
    """(deprecated) Use ``scitex-dataset general huggingface ...``"""
    click.echo(
        "error: `scitex-dataset hf ...` was renamed to "
        "`scitex-dataset general huggingface ...`.\n"
        "Re-run with: scitex-dataset general huggingface [search|info|fetch|download-file]",
        err=True,
    )
    ctx.exit(2)


# §1a: install-shell-completion + print-shell-completion (canonical leaves).
# Replaces the legacy `completion` and `print-tab-completion` commands; the
# attach helper also registers hidden deprecated aliases for those names.
try:
    from scitex_dev._cli._completion import attach_shell_completion

    attach_shell_completion(main, prog_name="scitex-dataset")
except ImportError:
    pass


if __name__ == "__main__":
    main()

# EOF


# audit §4 — inject version into root --help
try:
    from importlib.metadata import version as _v
    main.help = (
        f"scitex-dataset (v{_v('scitex-dataset')}) — "
        + (main.help or "").lstrip()
    )
except Exception:
    pass
