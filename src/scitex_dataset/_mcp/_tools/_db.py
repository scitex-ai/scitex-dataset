#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_db.py

"""Dataset-index MCP tools (build, search, stats)."""

from typing import Any, Dict, List, Optional


def register_db_tools(mcp) -> None:
    """Register ``db_build / db_search / db_show_stats``."""

    @mcp.tool()
    def db_build(
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build or rebuild the full-text dataset index across catalog sources - use whenever the user asks to populate search, refresh the scitex-dataset index, or pre-warm before bulk querying. Default sources: all 10 catalog sources (HuggingFace is on-demand and excluded from the index by design). The index lives in the shared SciTeX store, so every host and agent reads one catalogue rather than each keeping a private copy."""
        from ... import database

        counts = database.build(sources=sources)
        return {
            "success": True,
            "indexed": counts,
            "total": sum(counts.values()),
            "store": database.store_description(),
        }

    @mcp.tool()
    def db_search(
        query: Optional[str] = None,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        min_subjects: Optional[int] = None,
        max_subjects: Optional[int] = None,
        min_downloads: Optional[int] = None,
        has_readme: bool = False,
        limit: int = 20,
        order_by: str = "downloads",
    ) -> List[Dict[str, Any]]:
        """Search the dataset index — full-text query plus structured filters (source, modality, subject range, downloads, readme presence). Requires ``db_build`` first. Query syntax: bare words are ANDed, "quoted phrases" are phrases, ``or`` disjoins, a leading ``-`` negates. ``source`` accepts any of the 10 catalog sources (openneuro, dandi, physionet, zenodo, figshare, openml, geo, chembl, moleculenet, clinicaltrials). For HuggingFace, use ``huggingface_search`` (live API)."""
        from ... import database

        return database.search(
            query=query,
            source=source,
            modality=modality,
            min_subjects=min_subjects,
            max_subjects=max_subjects,
            min_downloads=min_downloads,
            has_readme=has_readme,
            limit=limit,
            order_by=order_by,
        )

    @mcp.tool()
    def db_show_stats() -> Dict[str, Any]:
        """Report dataset-index health — per-source counts, total rows, which store holds the index, and when it was last written. Use whenever the user asks "how many datasets are indexed?", "is the index fresh?", or is diagnosing an empty ``db_search``."""
        from ... import database

        return database.get_stats()


__all__ = ["register_db_tools"]

# EOF
