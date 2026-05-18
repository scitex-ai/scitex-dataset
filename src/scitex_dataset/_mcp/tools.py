#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/tools.py

"""Public re-export of ``register_all_tools`` for external aggregator servers.

Tool definitions live in ``_mcp/_tools/`` (one module per logical group);
this file is the stable import path used by scitex-python's unified MCP
server.
"""

from ._tools import register_all_tools

__all__ = ["register_all_tools"]

# EOF
