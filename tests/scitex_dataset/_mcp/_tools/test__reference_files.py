"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_reference_files.py``."""

from __future__ import annotations


class _RecordingMCP:
    """Hand-rolled MCP server stand-in: records registered tool callables."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        del a, k

        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _registered_tools():
    from scitex_dataset._mcp._tools._reference_files import register_reference_tools

    m = _RecordingMCP()
    register_reference_tools(m)
    return m.tools


def test_register_reference_tools_attaches_three_tools():
    # Arrange
    expected = {"fetch_file", "fetch_reference", "list_references"}
    # Act
    names = set(_registered_tools())
    # Assert
    assert names == expected


def test_fetch_file_tool_returns_record_dict(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    tool = _registered_tools()["fetch_file"]
    # Act
    out = tool(f"{base}/payload.bin", loopback_payload_sha256, str(tmp_path))
    # Assert
    assert (out["sha256"], out["cached"]) == (loopback_payload_sha256, False)


def test_list_references_tool_includes_mnist():
    # Arrange
    tool = _registered_tools()["list_references"]
    # Act
    ids = [e["id"] for e in tool()]
    # Assert
    assert "mnist" in ids


# EOF
