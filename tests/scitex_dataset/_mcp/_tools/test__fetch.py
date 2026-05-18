"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_fetch.py``."""

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


def test_register_fetch_tools_attaches_ten_catalog_tools():
    # Arrange
    from scitex_dataset._mcp._tools._fetch import register_fetch_tools

    m = _MockMCP()
    # Act
    register_fetch_tools(m)
    # Assert
    assert len(m.tools) == 10


def test_register_fetch_tools_attaches_only_fetch_suffixed_names():
    # Arrange
    from scitex_dataset._mcp._tools._fetch import register_fetch_tools

    m = _MockMCP()
    # Act
    register_fetch_tools(m)
    # Assert
    assert all(name.endswith("_fetch") for name in m.tools)
