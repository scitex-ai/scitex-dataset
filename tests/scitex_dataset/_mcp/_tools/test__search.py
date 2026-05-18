"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_search.py``."""

from __future__ import annotations


class _MockMCP:
    """Hand-rolled MCP server stub used for tool registration assertions."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        del a, k

        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def test_register_search_tools_attaches_filter_results():
    # Arrange
    from scitex_dataset._mcp._tools._search import register_search_tools

    m = _MockMCP()
    # Act
    register_search_tools(m)
    # Assert
    assert "filter_results" in m.tools


def test_register_search_tools_attaches_list_sources():
    # Arrange
    from scitex_dataset._mcp._tools._search import register_search_tools

    m = _MockMCP()
    # Act
    register_search_tools(m)
    # Assert
    assert "list_sources" in m.tools


def test_dataset_list_sources_returns_count_equal_to_all_sources():
    # Arrange
    from scitex_dataset._mcp._tools._search import register_search_tools
    from scitex_dataset._sources import ALL_SOURCES

    m = _MockMCP()
    register_search_tools(m)
    # Act
    result = m.tools["list_sources"]()
    # Assert
    assert result["count"] == len(ALL_SOURCES)
