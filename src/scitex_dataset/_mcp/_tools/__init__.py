#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/__init__.py

"""MCP tool registration for scitex-dataset.

Single entry point: ``register_all_tools(mcp)``. Used by the standalone
server (``_mcp/server.py``) and by external aggregator servers via
``_mcp/tools.py``.
"""

from ._ai_for_science import register_ai_for_science_tools
from ._db import register_db_tools
from ._fetch import register_fetch_tools
from ._hf import register_hf_tools
from ._search import register_search_tools
from ._skills import register_skills_tools


def register_all_tools(mcp) -> None:
    """Register every scitex-dataset MCP tool on the given FastMCP server."""
    register_fetch_tools(mcp)
    register_search_tools(mcp)
    register_db_tools(mcp)
    register_hf_tools(mcp)
    register_ai_for_science_tools(mcp)
    register_skills_tools(mcp)


__all__ = ["register_all_tools"]

# EOF
