#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_skills.py

"""Skills introspection MCP tools (per the §5 audit-mcp-tools convention)."""

import json
from pathlib import Path


def register_skills_tools(mcp) -> None:
    """Register ``dataset_skills_list`` and ``dataset_skills_get``."""

    skills_dir = Path(__file__).resolve().parents[2] / "_skills" / "scitex-dataset"

    @mcp.tool()
    def dataset_skills_list() -> str:
        """List the names of every skill page shipped by scitex-dataset.

        Returns
        -------
        str
            JSON string with ``{"success": true, "package": "scitex-dataset", "skills": [...]}``.
        """
        try:
            names = sorted(
                p.stem for p in skills_dir.glob("*.md") if p.name != "SKILL.md"
            )
            return json.dumps(
                {"success": True, "package": "scitex-dataset", "skills": names},
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    def dataset_skills_get(name: str) -> str:
        """Fetch the full Markdown content of one scitex-dataset skill page."""
        try:
            target = skills_dir / f"{name}.md"
            if not target.exists():
                available = sorted(
                    p.stem for p in skills_dir.glob("*.md") if p.name != "SKILL.md"
                )
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
                    "content": target.read_text(encoding="utf-8"),
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)


__all__ = ["register_skills_tools"]

# EOF
