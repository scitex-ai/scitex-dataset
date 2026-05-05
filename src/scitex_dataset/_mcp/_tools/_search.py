#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_search.py

"""Cross-source search + source-listing MCP tools."""

from typing import Any, Dict, List, Optional


def register_search_tools(mcp) -> None:
    """Register ``dataset_search`` and ``dataset_list_sources``."""

    @mcp.tool()
    def dataset_search(
        datasets: List[Dict[str, Any]],
        modality: Optional[str] = None,
        min_subjects: Optional[int] = None,
        max_subjects: Optional[int] = None,
        task_contains: Optional[str] = None,
        text_query: Optional[str] = None,
        min_downloads: Optional[int] = None,
        has_readme: bool = False,
        sort_by: str = "downloads",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Filter and rank a list of normalized dataset dicts (from any catalog source) by modality / subjects / task / full-text / downloads / readme presence. Use after any ``dataset_<src>_fetch`` call. For offline querying without passing dataset lists back-and-forth, prefer ``dataset_db_search``."""
        from ...search import search_datasets, sort_datasets

        results = search_datasets(
            datasets,
            modality=modality,
            min_subjects=min_subjects,
            max_subjects=max_subjects,
            task_contains=task_contains,
            text_query=text_query,
            min_downloads=min_downloads,
            has_readme=has_readme,
        )
        results = sort_datasets(results, by=sort_by, descending=True)
        return results[:limit]

    @mcp.tool()
    def dataset_list_sources() -> Dict[str, Any]:
        """List every dataset repository scitex-dataset supports - use whenever the user asks "what sources are available?", "which scientific data hubs are covered?", or before calling a per-source fetcher. Returns the 11 supported sources (10 catalog + 1 on-demand HuggingFace) with URLs, formats, domain tags, and ``kind`` (``catalog`` vs ``ondemand``)."""
        from ..._sources import SOURCE_INFO

        return {"sources": SOURCE_INFO, "count": len(SOURCE_INFO)}


__all__ = ["register_search_tools"]

# EOF
