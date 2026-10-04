#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/server.py

"""MCP server for scitex-dataset.

Tool definitions live in ``_mcp/_tools/`` and are registered via
``register_all_tools(mcp)``. This module is a thin shim: FastMCP init,
tool registration, resource endpoint, and the ``__main__`` entry.

Usage::

    fastmcp run scitex_dataset._mcp.server:mcp
    # or
    scitex-dataset mcp start
"""

from pathlib import Path

try:
    from fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "fastmcp not installed. MCP support needs it: "
        "pip install scitex-dataset[mcp]"
    ) from exc

from .._branding import get_mcp_instructions, get_mcp_server_name
from ._tools import register_all_tools

mcp = FastMCP(
    name=get_mcp_server_name(),
    instructions=get_mcp_instructions(),
)

register_all_tools(mcp)


@mcp.resource("scitex-dataset://readme")
def get_readme() -> str:
    """Get package README."""
    readme_path = Path(__file__).parent.parent.parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    return "scitex-dataset: Unified interface for scientific dataset discovery."


if __name__ == "__main__":
    mcp.run()

# EOF
