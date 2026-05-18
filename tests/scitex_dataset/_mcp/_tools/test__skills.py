"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_skills.py``."""

from __future__ import annotations

import json


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


def test_skills_list_payload_marks_success_true():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    listing = json.loads(m.tools["skills_list"]())
    # Assert
    assert listing["success"] is True


def test_skills_list_contains_installation_page():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    listing = json.loads(m.tools["skills_list"]())
    # Assert
    assert "01_installation" in listing["skills"]


def test_skills_list_contains_cli_reference_page():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    listing = json.loads(m.tools["skills_list"]())
    # Assert
    assert "04_cli-reference" in listing["skills"]


def test_skills_get_returns_success_true_for_cli_reference():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    payload = json.loads(m.tools["skills_get"]("04_cli-reference"))
    # Assert
    assert payload["success"] is True


def test_skills_get_returns_requested_skill_name_for_cli_reference():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    payload = json.loads(m.tools["skills_get"]("04_cli-reference"))
    # Assert
    assert payload["name"] == "04_cli-reference"


def test_skills_get_returns_content_mentioning_domain_for_cli_reference():
    # Arrange
    from scitex_dataset._mcp._tools._skills import register_skills_tools

    m = _MockMCP()
    register_skills_tools(m)
    # Act
    payload = json.loads(m.tools["skills_get"]("04_cli-reference"))
    # Assert
    assert "domain" in payload["content"]
