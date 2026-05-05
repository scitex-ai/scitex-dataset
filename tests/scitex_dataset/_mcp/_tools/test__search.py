"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_search.py``."""

from __future__ import annotations


class _M:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        del a, k

        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def test_register_search_tools_attaches_filter_and_list_sources():
    from scitex_dataset._mcp._tools._search import register_search_tools

    m = _M()
    register_search_tools(m)
    assert "filter_results" in m.tools
    assert "list_sources" in m.tools


def test_dataset_list_sources_returns_eleven():
    from scitex_dataset._mcp._tools._search import register_search_tools
    from scitex_dataset._sources import ALL_SOURCES

    m = _M()
    register_search_tools(m)
    result = m.tools["list_sources"]()
    assert result["count"] == len(ALL_SOURCES)
