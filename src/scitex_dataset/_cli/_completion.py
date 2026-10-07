#!/usr/bin/env python3
# File: src/scitex_dataset/_cli/_completion.py
"""Shell tab-completion drop-in (fleet standard v1).

``scitex-dataset completion install`` writes the click-generated
completion script to the canonical drop-in::

    $SCITEX_DIR/dataset/runtime/completion/scitex-dataset

(``$SCITEX_DIR`` defaults to ``~/.scitex``) atomically and
idempotently, then prints the path. Nothing is ever appended to shell
rc files — the user's shell framework sources the drop-in instead.

``scitex-dataset completion status`` checks the drop-in.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import click

logger = logging.getLogger(__name__)

PROG_NAME = "scitex-dataset"
SHORT = "dataset"
COMPLETE_VAR = "_SCITEX_DATASET_COMPLETE"
_VALID_SHELLS = ("bash", "zsh", "fish")


def _scitex_dir() -> Path:
    """Resolve ``$SCITEX_DIR`` (default ``~/.scitex``) at call time."""
    raw = os.environ.get("SCITEX_DIR") or os.path.expanduser("~/.scitex")
    return Path(os.path.expanduser(raw))


def dropin_path() -> Path:
    """Canonical drop-in path (resolved at call time for HOME isolation)."""
    return _scitex_dir() / SHORT / "runtime" / "completion" / PROG_NAME


def generate_script(shell: str) -> str:
    """Return the click-generated completion script for ``shell``."""
    from click.shell_completion import get_completion_class

    from scitex_dataset._cli import main as root_cli

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"unsupported shell: {shell}")
    comp = comp_cls(root_cli, {}, PROG_NAME, COMPLETE_VAR)
    script = comp.source().strip()
    if not script:
        raise click.ClickException(
            f"Failed to generate {shell} completion script for {PROG_NAME}."
        )
    return script + "\n"


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@click.group("completion", invoke_without_command=True)
@click.pass_context
def completion_group(ctx):
    """Shell tab-completion management (drop-in, no rc edits).

    \b
    Examples:
        scitex-dataset completion install
        scitex-dataset completion install --shell bash
        scitex-dataset completion status
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@completion_group.command("install")
@click.option(
    "--shell",
    type=click.Choice(list(_VALID_SHELLS)),
    default="bash",
    show_default=True,
    help="Target shell.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Accepted for automation; install never prompts.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the drop-in path without writing.",
)
def completion_install(shell: str, yes: bool, dry_run: bool) -> None:
    """Write/refresh the completion drop-in, then print its path.

    \b
    Examples:
        scitex-dataset completion install
        scitex-dataset completion install --shell bash
        scitex-dataset completion install --shell zsh --yes
    """
    del yes  # accepted for automation; install never prompts.
    target = dropin_path()
    if dry_run:
        click.echo(str(target))
        return
    script = generate_script(shell)
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    if existing != script:
        _atomic_write(target, script)
        logger.debug("completion drop-in written: %s", target)
    else:
        logger.debug("completion drop-in already current: %s", target)
    click.echo(str(target))


@completion_group.command("status")
@click.option(
    "--shell",
    type=click.Choice(list(_VALID_SHELLS)),
    default="bash",
    show_default=True,
    help="Target shell (drop-in path is shell-independent).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON.",
)
def completion_status(shell: str, as_json: bool) -> None:
    """Check whether the completion drop-in is installed.

    \b
    Examples:
        scitex-dataset completion status
        scitex-dataset completion status --json
    """
    del shell  # path is shell-independent; accepted for symmetry.
    target = dropin_path()
    installed = target.is_file() and target.stat().st_size > 0
    if as_json:
        import json as _json

        click.echo(
            _json.dumps(
                {"installed": installed, "path": str(target), "prog": PROG_NAME},
                indent=2,
            )
        )
        return
    if installed:
        click.echo(f"installed: {target}")
    else:
        click.echo(f"not installed: {target}")
        click.echo("Install with: scitex-dataset completion install")


@click.command("print-shell-completion")
@click.option(
    "--shell",
    type=click.Choice(list(_VALID_SHELLS)),
    default="bash",
    show_default=True,
    help="Target shell.",
)
def print_shell_completion_cmd(shell: str) -> None:
    """Print the completion script to stdout (no filesystem changes).

    \b
    Example:
        eval "$(scitex-dataset print-shell-completion --shell bash)"
    """
    click.echo(generate_script(shell), nl=False)


@click.command("install-shell-completion")
@click.option(
    "--shell",
    type=click.Choice(list(_VALID_SHELLS)),
    default="bash",
    show_default=True,
    help="Target shell.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Accepted for automation; install never prompts.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the drop-in path without writing.",
)
def install_shell_completion_cmd(shell: str, yes: bool, dry_run: bool) -> None:
    """Install tab-completion via the drop-in (never edits shell rc files).

    Canonical spelling is ``scitex-dataset completion install``; this leaf
    exists for audit compatibility and behaves identically.

    \b
    Example:
        scitex-dataset install-shell-completion --shell bash
    """
    del yes
    target = dropin_path()
    if dry_run:
        click.echo(str(target))
        return
    script = generate_script(shell)
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    if existing != script:
        _atomic_write(target, script)
        logger.debug("completion drop-in written: %s", target)
    click.echo(str(target))


def register_completion_commands(group: click.Group) -> None:
    """Attach the drop-in ``completion`` group + audit leaves to ``group``."""
    group.add_command(completion_group)
    group.add_command(print_shell_completion_cmd)
    group.add_command(install_shell_completion_cmd)


# EOF
