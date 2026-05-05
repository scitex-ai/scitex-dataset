"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_db.py``."""

from __future__ import annotations


def test_register_db_tools_attaches_three():
    from scitex_dataset._mcp._tools._db import register_db_tools

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
    register_db_tools(m)
    assert set(m.tools) == {"dataset_db_build", "dataset_db_search", "dataset_db_stats"}
