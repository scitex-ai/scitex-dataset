"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_fetch.py``."""

from __future__ import annotations


def test_register_fetch_tools_attaches_ten_catalog_tools():
    from scitex_dataset._mcp._tools._fetch import register_fetch_tools

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
    register_fetch_tools(m)
    assert len(m.tools) == 10
    for name in m.tools:
        assert name.endswith("_fetch")
