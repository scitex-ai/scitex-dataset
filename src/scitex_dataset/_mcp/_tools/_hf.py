#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_hf.py

"""HuggingFace on-demand MCP tools."""

from typing import Any, Dict, List, Optional


def register_hf_tools(mcp) -> None:
    """Register ``dataset_hf_*`` tools (alias-named ``dataset_huggingface_*`` also accepted)."""

    @mcp.tool()
    def hf_fetch(
        repo_id: str,
        local_dir: Optional[str] = None,
        repo_type: str = "dataset",
        max_workers: int = 4,
        hf_home_override: Optional[str] = None,
    ) -> str:
        """Fetch a complete HuggingFace dataset or model to disk - use whenever the user asks to download an HF dataset (gated or public), access Anthropic/BioMysteryBench-full or other large models/datasets, or cache HF content to project filesystem (especially on Spartan with HF_HOME override). Drop-in replacement for huggingface_hub.snapshot_download with smart project-FS routing."""
        from ...general.huggingface import fetch_dataset

        result_path = fetch_dataset(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            max_workers=max_workers,
            hf_home_override=hf_home_override,
        )
        return str(result_path)

    @mcp.tool()
    def hf_search(
        query: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search the HuggingFace Hub for datasets by keyword - use whenever the user asks to find or browse HF datasets by query, discover public ML benchmarks, or search for domain-specific datasets (biology, medical, etc.). Drop-in replacement for HuggingFace Hub web search or huggingface_hub.list_datasets()."""
        from ...general.huggingface import search_hub

        return search_hub(query=query, limit=limit)

    @mcp.tool()
    def hf_info(
        repo_id: str,
        repo_type: str = "dataset",
    ) -> Dict[str, Any]:
        """Get metadata about a HuggingFace dataset or model - use whenever the user asks for info about a specific HF repo, check license / size / gating, or verify before downloading. Drop-in replacement for huggingface_hub.dataset_info() / model_info()."""
        from ...general.huggingface import dataset_info

        return dataset_info(repo_id=repo_id, repo_type=repo_type)

    @mcp.tool()
    def hf_download_file(
        repo_id: str,
        filename: str,
        local_dir: Optional[str] = None,
        repo_type: str = "dataset",
    ) -> str:
        """Download a single file from a HuggingFace repository - use when the user wants just one file (README.md, metadata.json, single data file) without the full snapshot. Drop-in replacement for huggingface_hub.hf_hub_download with project-FS awareness."""
        from ...general.huggingface import download_file

        result_path = download_file(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type=repo_type,
        )
        return str(result_path)


__all__ = ["register_hf_tools"]

# EOF
