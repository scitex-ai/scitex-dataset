#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-30 10:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/_cli/_mcp_commands.py

"""MCP-related CLI commands."""

import json

import click

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
@click.option("--help-recursive", is_flag=True, help="Show help for all commands.")
@click.pass_context
def mcp(ctx: click.Context, help_recursive: bool):
    """MCP (Model Context Protocol) server commands."""
    if help_recursive:
        click.echo(mcp.get_help(ctx))
        for name, cmd in sorted(mcp.commands.items()):
            _print_command_help(cmd, f"scitex-dataset mcp {name}", ctx)
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp.command("start")
@click.option("--dry-run", is_flag=True, help="Print launch plan without starting.")
@click.option(
    "-y", "--yes", is_flag=True, help="Suppress interactive confirmation (assume yes)."
)
def mcp_start(dry_run: bool, yes: bool) -> None:
    """Start the scitex-dataset MCP server.

    \b
    Example:
      $ scitex-dataset mcp start
      $ scitex-dataset mcp start --dry-run
    """
    if dry_run:
        click.echo("DRY RUN — would start scitex-dataset MCP server (stdio transport)")
        return

    try:
        from .._mcp.server import mcp as mcp_server
    except ImportError as e:
        raise click.ClickException(
            f"fastmcp not installed. Install: pip install scitex-dataset[mcp]\n{e}"
        ) from e

    click.echo("Starting scitex-dataset MCP server...")
    mcp_server.run()


def _extract_return_keys(description: str) -> list:
    """Extract return dict keys from docstring Returns section."""
    import re

    if not description or "Returns" not in description:
        return []
    match = re.search(
        r"Returns\s*[-]+\s*\w+\s*(.+?)(?:Raises|Examples|Notes|\Z)",
        description,
        re.DOTALL,
    )
    if not match:
        return []
    return re.findall(r"'([a-z_]+)'", match.group(1))


def _format_signature(tool, multiline: bool = False, indent: str = "  ") -> str:
    """Format tool as Python-like function signature with colors."""
    import inspect

    params = []
    if hasattr(tool, "parameters") and tool.parameters:
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for name, info in props.items():
            ptype = info.get("type", "any")
            default = info.get("default")
            name_s = click.style(name, fg="white", bold=True)
            type_s = click.style(ptype, fg="cyan")
            if name in required:
                p = f"{name_s}: {type_s}"
            elif default is not None:
                def_str = repr(default) if len(repr(default)) < 20 else "..."
                p = f"{name_s}: {type_s} = {click.style(def_str, fg='yellow')}"
            else:
                p = f"{name_s}: {type_s} = {click.style('None', fg='yellow')}"
            params.append(p)

    # Get return type from function annotation + dict keys from docstring
    ret_type = ""
    if hasattr(tool, "fn") and tool.fn:
        try:
            sig = inspect.signature(tool.fn)
            if sig.return_annotation != inspect.Parameter.empty:
                ret = sig.return_annotation
                ret_name = ret.__name__ if hasattr(ret, "__name__") else str(ret)
                keys = (
                    _extract_return_keys(tool.description) if tool.description else []
                )
                keys_str = (
                    click.style(f"{{{', '.join(keys)}}}", fg="yellow") if keys else ""
                )
                ret_type = f" -> {click.style(ret_name, fg='magenta')}{keys_str}"
        except Exception:
            pass

    name_s = click.style(tool.name, fg="green", bold=True)
    if multiline and len(params) > 2:
        param_indent = indent + "    "
        params_str = ",\n".join(f"{param_indent}{p}" for p in params)
        return f"{indent}{name_s}(\n{params_str}\n{indent}){ret_type}"
    return f"{indent}{name_s}({', '.join(params)}){ret_type}"


def _estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~4 chars per token)."""
    return len(text) // 4 if text else 0


def _get_mcp_summary(mcp_server) -> dict:
    """Get MCP server summary statistics."""
    from scitex_dev import get_tools_sync

    tools = list(get_tools_sync(mcp_server).values())

    # Calculate total context size
    instructions = getattr(mcp_server, "instructions", "") or ""
    total_desc = sum(len(t.description or "") for t in tools)
    total_params = sum(
        len(json.dumps(t.parameters))
        if hasattr(t, "parameters") and t.parameters
        else 0
        for t in tools
    )

    return {
        "name": getattr(mcp_server, "name", "unknown"),
        "tool_count": len(tools),
        "instructions_chars": len(instructions),
        "instructions_tokens": _estimate_tokens(instructions),
        "descriptions_chars": total_desc,
        "descriptions_tokens": _estimate_tokens("x" * total_desc),
        "schemas_chars": total_params,
        "schemas_tokens": _estimate_tokens("x" * total_params),
        "total_context_tokens": (
            _estimate_tokens(instructions)
            + _estimate_tokens("x" * total_desc)
            + _estimate_tokens("x" * total_params)
        ),
    }


@mcp.command("list-tools")
@click.option("-v", "--verbose", count=True, help="Verbosity: -v, -vv, -vvv.")
@click.option("-c", "--compact", is_flag=True, help="Compact signatures (single line)")
@click.option(
    "--module", "-m", type=str, default=None, help="Filter by module (dataset, db)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--summary", "show_summary", is_flag=True, help="Show context summary.")
def mcp_list_tools(
    verbose: int, compact: bool, module: str, as_json: bool, show_summary: bool
) -> None:
    """List available MCP tools.

    Verbosity: (none) names, -v signatures, -vv +description, -vvv full.
    Signatures are expanded by default; use -c/--compact for single line.

    \b
    Example:
      $ scitex-dataset mcp list-tools
      $ scitex-dataset mcp list-tools -vv
      $ scitex-dataset mcp list-tools --json
    """
    try:
        from .._mcp.server import mcp as mcp_server
    except ImportError as e:
        raise click.ClickException(
            f"fastmcp not installed. Install: pip install scitex-dataset[mcp]\n{e}"
        ) from e

    # Get all tools
    from scitex_dev import get_tools_sync

    _tools_map = get_tools_sync(mcp_server)
    tools = list(_tools_map.keys())

    # Group by module prefix
    modules = {}
    for tool in sorted(tools):
        prefix = tool.split("_")[0]
        if prefix not in modules:
            modules[prefix] = []
        modules[prefix].append(tool)

    # Filter by module if specified
    if module:
        module = module.lower()
        if module not in modules:
            click.secho(f"ERROR: Unknown module '{module}'", fg="red", err=True)
            click.echo(f"Available modules: {', '.join(sorted(modules.keys()))}")
            raise SystemExit(1)
        modules = {module: modules[module]}

    summary = _get_mcp_summary(mcp_server)

    if as_json:
        output = {
            "summary": summary,
            "total": sum(len(t) for t in modules.values()),
            "modules": {},
        }
        for mod, tool_list in modules.items():
            output["modules"][mod] = {"count": len(tool_list), "tools": []}
            for tool_name in tool_list:
                tool_obj = _tools_map.get(tool_name)
                schema = tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
                output["modules"][mod]["tools"].append(
                    {
                        "name": tool_name,
                        "signature": _format_signature(tool_obj)
                        if tool_obj
                        else tool_name,
                        "description": tool_obj.description if tool_obj else "",
                        "parameters": schema,
                    }
                )
        click.echo(json.dumps(output, indent=2))
        return

    # Header
    total = sum(len(t) for t in modules.values())
    click.secho(f"scitex-dataset MCP: {summary['name']}", fg="cyan", bold=True)
    click.echo(f"Tools: {total} ({len(modules)} modules)")
    if show_summary:
        click.echo(f"Context: ~{summary['total_context_tokens']:,} tokens")
        click.echo(f"  Instructions: ~{summary['instructions_tokens']:,} tokens")
        click.echo(f"  Descriptions: ~{summary['descriptions_tokens']:,} tokens")
        click.echo(f"  Schemas: ~{summary['schemas_tokens']:,} tokens")
    click.echo()

    for mod, tool_list in sorted(modules.items()):
        click.secho(f"{mod}: {len(tool_list)} tools", fg="green", bold=True)
        for tool_name in tool_list:
            tool_obj = _tools_map.get(tool_name)

            if verbose == 0:
                # Names only
                click.echo(f"  {tool_name}")
            elif verbose == 1:
                # Full signature
                sig = (
                    _format_signature(tool_obj, multiline=not compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                click.echo(sig)
            elif verbose == 2:
                # Signature + one-line description
                sig = (
                    _format_signature(tool_obj, multiline=not compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                click.echo(sig)
                if tool_obj and tool_obj.description:
                    desc = tool_obj.description.split("\n")[0].strip()
                    click.echo(f"    {desc}")
                click.echo()
            else:
                # Signature + full description
                sig = (
                    _format_signature(tool_obj, multiline=not compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                click.echo(sig)
                if tool_obj and tool_obj.description:
                    for line in tool_obj.description.strip().split("\n"):
                        click.echo(f"    {line}")
                click.echo()
        click.echo()


@mcp.command("doctor")
def mcp_doctor() -> None:
    """Check MCP server dependencies.

    \b
    Example:
      $ scitex-dataset mcp doctor
    """
    click.secho("Checking MCP dependencies...", fg="cyan")

    try:
        import fastmcp

        click.secho("  OK ", fg="green", nl=False)
        click.echo(f"fastmcp {fastmcp.__version__}")
    except ImportError:
        click.secho("  NG ", fg="red", nl=False)
        click.echo("fastmcp not installed")
        click.echo("     Install: pip install scitex-dataset[mcp]")
        return

    try:
        from .._mcp.server import mcp as mcp_server

        click.secho("  OK ", fg="green", nl=False)
        from scitex_dev import get_tools_sync

        click.echo(f"MCP server ({len(get_tools_sync(mcp_server))} tools)")
    except Exception as exc:
        click.secho("  NG ", fg="red", nl=False)
        click.echo(f"MCP server error: {exc}")
        return

    click.secho("\nMCP server ready.", fg="green")
    click.echo("Run: scitex-dataset mcp start")


@mcp.command("install")
@click.option("--claude-code", is_flag=True, help="Show Claude Code config.")
@click.option("--dry-run", is_flag=True, help="Print install plan without executing.")
@click.option(
    "-y", "--yes", is_flag=True, help="Suppress interactive confirmation (assume yes)."
)
def mcp_install(claude_code: bool, dry_run: bool, yes: bool) -> None:
    """Show MCP installation instructions.

    \b
    Example:
      $ scitex-dataset mcp install
      $ scitex-dataset mcp install --claude-code
    """
    if dry_run:
        click.echo(
            "DRY RUN — would print MCP install instructions "
            f"(claude_code={claude_code})"
        )
        return

    if claude_code:
        click.secho("Add to Claude Code MCP config:", fg="cyan")
        click.echo()
        click.echo('  "scitex-dataset": {')
        click.echo('    "command": "scitex-dataset",')
        click.echo('    "args": ["mcp", "start"]')
        click.echo("  }")
    else:
        click.secho("scitex-dataset MCP Server Installation", fg="cyan", bold=True)
        click.echo("=" * 40)
        click.echo()
        click.echo("1. Install: pip install scitex-dataset[mcp]")
        click.echo("2. Config:  scitex-dataset mcp install --claude-code")
        click.echo("3. Test:    scitex-dataset mcp doctor")


# EOF
