#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_skills.py

"""Skills introspection MCP tools (per the §5 audit-mcp-tools convention)."""

import json
from ..._skill_resources import _read_skill_page, _skill_names


def register_skills_tools(mcp) -> None:
    """Register ``skills_list`` and ``skills_get``."""

    @mcp.tool()
    def skills_list() -> str:
        """List the names of every skill page shipped by scitex-dataset.

        Returns
        -------
        str
            JSON string with ``{"success": true, "package": "scitex-dataset", "skills": [...]}``.
        """
        try:
            names = _skill_names()
            return json.dumps(
                {"success": True, "package": "scitex-dataset", "skills": names},
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    def skills_get(name: str) -> str:
        """Fetch the full Markdown content of one scitex-dataset skill page."""
        try:
            content = _read_skill_page(name)
            if content is None:
                available = _skill_names()
                return json.dumps(
                    {
                        "success": False,
                        "error": f"unknown skill {name!r}; available: {available}",
                    },
                    indent=2,
                )
            return json.dumps(
                {
                    "success": True,
                    "package": "scitex-dataset",
                    "name": name,
                    "content": content,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)


__all__ = ["register_skills_tools"]

# EOF
