"""Mirror test for ``src/scitex_dataset/_mcp/_tools/_hf.py``."""

from __future__ import annotations


def test_register_hf_tools_attaches_four():
    from scitex_dataset._mcp._tools._hf import register_hf_tools

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
    register_hf_tools(m)
    assert set(m.tools) == {
        "huggingface_fetch",
        "huggingface_search",
        "huggingface_info",
        "huggingface_download_file",
    }
