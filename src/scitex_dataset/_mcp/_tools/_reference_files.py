#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_reference_files.py

"""Checksum-verified file fetch + reference-file registry MCP tools."""

from typing import Any, Dict, List, Optional


def register_reference_tools(mcp) -> None:
    """Register ``fetch_file`` / ``fetch_reference`` / ``list_references``."""

    @mcp.tool()
    def fetch_file(
        url: str,
        sha256: str,
        dest_dir: str,
        filename: Optional[str] = None,
        timeout: float = 60.0,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Download one file by URL, verify its sha256, cache it and return a provenance record (url, sha256, size_bytes, path, retrieved_at, cached) - use whenever the user needs a specific data file fetched reproducibly with integrity checking. A verified existing copy is reused; a checksum mismatch raises and the partial file is removed. Drop-in replacement for wget/curl + manual sha256sum, pooch.retrieve, or keras.utils.get_file."""
        from ..._fetch_file import fetch_file as _fetch_file

        return _fetch_file(
            url,
            sha256,
            dest_dir,
            filename,
            timeout=timeout,
            replace=replace,
        ).to_dict()

    @mcp.tool()
    def fetch_reference(
        name: str,
        dest_dir: str,
        timeout: float = 60.0,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a well-known reference dataset file (e.g. "mnist") pinned by sha256 and return its provenance record including license and citation - use whenever the user asks for MNIST or another registered reference dataset with provenance. Drop-in replacement for keras.datasets.mnist.load_data() / torchvision.datasets.MNIST downloads."""
        from ...general.reference import fetch_reference as _fetch_reference

        return _fetch_reference(
            name, dest_dir, timeout=timeout, replace=replace
        ).to_dict()

    @mcp.tool()
    def list_references() -> List[Dict[str, Any]]:
        """List the registered reference files (id, url, sha256, size, license, citation) - use whenever the user asks which reference datasets (MNIST, ...) scitex-dataset can fetch with provenance."""
        from ...general.reference import list_references as _list_references

        return _list_references()


__all__ = ["register_reference_tools"]

# EOF
