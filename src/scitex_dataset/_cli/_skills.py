#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_skills.py

"""``scitex-dataset skills`` — list / get / install package skill pages.

Skill pages live under ``src/scitex_dataset/_skills/scitex-dataset/``
and are exposed by-id via this CLI plus the matching ``dataset_skills_*``
MCP tools (see ``_mcp/_tools/_skills.py``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _skills_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "_skills" / "scitex-dataset"


def _skill_names() -> list[str]:
    d = _skills_dir()
    return sorted(p.stem for p in d.glob("*.md") if p.name != "SKILL.md")


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def skills(ctx):
    """Browse / fetch / install bundled skill pages."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@skills.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit names as JSON.")
def skills_list(as_json: bool):
    """List bundled skill page names.

    \b
    Example:
      $ scitex-dataset skills list
      $ scitex-dataset skills list --json
    """
    names = _skill_names()
    if as_json:
        click.echo(json.dumps({"skills": names, "count": len(names)}, indent=2))
        return
    for name in names:
        click.echo(name)


@skills.command("get")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Emit content as JSON.")
def skills_get(name: str, as_json: bool):
    """Print the markdown content of one skill page.

    \b
    Example:
      $ scitex-dataset skills get 02_quick-start
      $ scitex-dataset skills get 04_cli-reference --json
    """
    target = _skills_dir() / f"{name}.md"
    if not target.is_file():
        if as_json:
            click.echo(
                json.dumps(
                    {
                        "success": False,
                        "error": f"unknown skill {name!r}",
                        "available": _skill_names(),
                    },
                    indent=2,
                )
            )
        else:
            click.echo(
                f"error: unknown skill {name!r}; available: {_skill_names()}",
                err=True,
            )
        raise SystemExit(2)
    content = target.read_text(encoding="utf-8")
    if as_json:
        click.echo(
            json.dumps({"success": True, "name": name, "content": content}, indent=2)
        )
        return
    click.echo(content)


@skills.command("install")
@click.option(
    "-d",
    "--dest",
    type=click.Path(),
    default=str(Path.home() / ".claude" / "skills" / "scitex" / "scitex-dataset"),
    show_default=True,
    help="Destination directory.",
)
@click.option("-f", "--force", is_flag=True, help="Overwrite if dest exists.")
@click.option("--dry-run", is_flag=True, help="Print plan without writing.")
@click.option("-y", "--yes", is_flag=True, help="Suppress interactive confirmation.")
def skills_install(dest: str, force: bool, dry_run: bool, yes: bool):
    """Install bundled skill pages to a target directory.

    \b
    Example:
      $ scitex-dataset skills install
      $ scitex-dataset skills install -d ./my-skills/dataset --force
      $ scitex-dataset skills install --dry-run
    """
    src = _skills_dir()
    dest_path = Path(dest).expanduser()
    pages = sorted(src.glob("*.md"))

    if dry_run:
        click.echo(f"DRY RUN — would install {len(pages)} skill pages to {dest_path}")
        for p in pages:
            click.echo(f"  {p.name}")
        return

    if dest_path.exists():
        if not force:
            click.echo(
                f"error: {dest_path} exists; pass --force to overwrite.",
                err=True,
            )
            raise SystemExit(2)
        if not yes:
            click.echo(
                f"error: {dest_path} exists; pass --yes/-y to overwrite.",
                err=True,
            )
            raise SystemExit(2)
        shutil.rmtree(dest_path)
    shutil.copytree(src, dest_path)
    click.echo(f"Installed {len(pages)} skill pages to {dest_path}")


__all__ = ["skills"]

# EOF
