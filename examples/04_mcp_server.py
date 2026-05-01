#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 04: Demonstrate scitex-dataset MCP tools.

Calls the same tools that `scitex-dataset mcp start` exposes,
directly as Python functions, and writes a sample payload to the
@stx.session output directory.

Usage:
    python 04_mcp_server.py
"""

import json
from pathlib import Path

import scitex as stx

from scitex_dataset._mcp.server import (
    dataset_list_sources,
    dataset_openneuro_fetch,
)


@stx.session
def main(
    CONFIG=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Exercise dataset_list_sources + dataset_openneuro_fetch MCP tools."""
    OUT = Path(CONFIG.SDIR_OUT)

    logger.info("=== MCP Tool: dataset_list_sources ===")
    sources = dataset_list_sources()
    for key, info in sources["sources"].items():
        logger.info(f"  {key}: {info['description']} ({info['domain']})")
    logger.info(f"  Total sources: {sources['count']}")

    logger.info("=== MCP Tool: dataset_openneuro_fetch ===")
    datasets = dataset_openneuro_fetch(max_datasets=5, batch_size=5)
    logger.info(f"  Fetched {len(datasets)} datasets")
    for ds in datasets:
        logger.info(f"  {ds['id']}: {ds['name'][:60]}")

    output_path = OUT / "mcp_demo_output.json"
    with open(output_path, "w") as f:
        json.dump({"sources": sources, "sample_datasets": datasets}, f, indent=2)
    logger.info(f"Saved to {output_path}")

    logger.info("=== Starting MCP Server ===")
    logger.info("  CLI:    scitex-dataset mcp start")
    logger.info("  Python: fastmcp run scitex_dataset._mcp.server:mcp")
    logger.info("  Tools:  scitex-dataset mcp list-tools")
    return 0


if __name__ == "__main__":
    main()

# EOF
