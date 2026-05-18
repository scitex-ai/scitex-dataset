"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_db.py``."""

from __future__ import annotations


class _MockMCP:
    """Hand-rolled MCP server stub (not a `unittest.mock` object).

    Captures decorator-registered tool functions in a dict so the test
    can assert on registration without spinning up a real FastMCP.
    """

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        del a, k

        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def test_register_db_tools_attaches_three_tools_named_db_prefix():
    # Arrange
    from scitex_dataset._mcp._tools._db import register_db_tools

    m = _MockMCP()
    # Act
    register_db_tools(m)
    # Assert
    assert set(m.tools) == {"db_build", "db_search", "db_show_stats"}
