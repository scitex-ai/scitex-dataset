"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_skills.py``."""

from __future__ import annotations

import json


def test_register_skills_tools_lists_real_pages():
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    class _M:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            del a, k

            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn

            return deco

    m = _M()
    register_skills_tools(m)

    listing = json.loads(m.tools["dataset_skills_list"]())
    assert listing["success"] is True
    assert "01_installation" in listing["skills"]
    assert "04_cli-reference" in listing["skills"]


def test_skills_get_round_trip():
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    class _M:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            del a, k

            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn

            return deco

    m = _M()
    register_skills_tools(m)
    payload = json.loads(m.tools["dataset_skills_get"]("04_cli-reference"))
    assert payload["success"] is True
    assert payload["name"] == "04_cli-reference"
    assert "domain" in payload["content"]
