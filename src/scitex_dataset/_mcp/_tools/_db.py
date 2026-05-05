#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_db.py

"""Local-database MCP tools (build, search, stats)."""

from typing import Any, Dict, List, Optional


def register_db_tools(mcp) -> None:
    """Register ``dataset_db_build / db_search / db_stats``."""

    @mcp.tool()
    def dataset_db_build(
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build or rebuild the local SQLite + FTS5 dataset index across catalog sources - use whenever the user asks to populate offline search, refresh the scitex-dataset cache, or pre-warm before bulk querying. Default sources: all 10 catalog sources (HuggingFace is on-demand and excluded from the index by design). Drop-in replacement for hand-rolled SQLite indexers over fetcher output."""
        from ... import database

        counts = database.build(sources=sources)
        stats = database.get_stats()
        return {
            "success": True,
            "indexed": counts,
            "total": sum(counts.values()),
            "database_path": stats.get("path"),
            "size_mb": stats.get("size_mb"),
        }

    @mcp.tool()
    def dataset_db_search(
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
        """Search the local SQLite + FTS5 dataset index — full-text query plus structured filters (source, modality, subject range, downloads, readme presence). Offline; requires ``dataset_db_build`` first. ``source`` accepts any of the 10 catalog sources (openneuro, dandi, physionet, zenodo, figshare, openml, geo, chembl, moleculenet, clinicaltrials). For HuggingFace, use ``dataset_hf_search`` (live API)."""
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
    def dataset_db_stats() -> Dict[str, Any]:
        """Report local dataset-index health — per-source counts, total rows, DB path, size on disk, last build time. Use whenever the user asks "how many datasets are indexed?", "is the cache fresh?", or is diagnosing an empty ``dataset_db_search``."""
        from ... import database

        return database.get_stats()


__all__ = ["register_db_tools"]

# EOF
