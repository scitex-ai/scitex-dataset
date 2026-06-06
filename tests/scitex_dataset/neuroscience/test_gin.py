#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/neuroscience/test_gin.py

"""Tests for the GIN (G-Node Infrastructure) dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``_httpx`` at the call site; we swap
``gin._httpx`` at the module namespace via a real save/restore
context manager. Filesystem fixtures are real temporary directories.
"""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import scitex_dataset.neuroscience.gin as _gin_mod
from scitex_dataset.neuroscience.gin import (
    GIN_API_V1,
    GIN_HOST,
    _parse_annex_pointer,
    _raw_url,
    _split_repo_id,
    format_dataset,
)


# --------------------------- httpx stub plumbing --------------------------- #

class _StubResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StubHttpx:
    """Captures ``_httpx.get`` calls; returns a FIFO queue of stub responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[tuple, dict]] = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self._responses:
            raise RuntimeError("StubHttpx ran out of queued responses")
        return self._responses.pop(0)


@contextmanager
def _swap_httpx(stub) -> Iterator[None]:
    saved = _gin_mod._httpx
    _gin_mod._httpx = stub  # type: ignore[assignment]
    try:
        yield
    finally:
        _gin_mod._httpx = saved  # type: ignore[assignment]


# --------------------------- constants + URLs --------------------------- #

def test_constants_point_at_public_gin_host():
    assert GIN_HOST == "https://gin.g-node.org"
    assert GIN_API_V1 == "https://gin.g-node.org/api/v1"


def test_split_repo_id_strips_trailing_dot_git_and_slashes():
    assert _split_repo_id("USZ_NCH/Human_MTL") == ("USZ_NCH", "Human_MTL")
    assert _split_repo_id("/USZ_NCH/Human_MTL/") == ("USZ_NCH", "Human_MTL")
    assert _split_repo_id("USZ_NCH/Human_MTL.git") == ("USZ_NCH", "Human_MTL")


def test_split_repo_id_rejects_malformed():
    import pytest
    with pytest.raises(ValueError):
        _split_repo_id("no-slash")
    with pytest.raises(ValueError):
        _split_repo_id("one/two/three")
    with pytest.raises(ValueError):
        _split_repo_id("")


def test_raw_url_encodes_path_segments():
    url = _raw_url("USZ_NCH/Human_MTL", "data_nix/Sub 01.h5", branch="master")
    assert url == (
        "https://gin.g-node.org/USZ_NCH/Human_MTL/raw/master/"
        "data_nix/Sub%2001.h5"
    )


# --------------------------- pointer parser --------------------------- #

def test_parse_annex_pointer_recognises_md5_backend():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ptr = td / "ptr.h5"
        ptr.write_text(
            "/annex/objects/MD5-s410499073--c43a9696fc48bb8a477135a53105e9d2\n"
        )
        got = _parse_annex_pointer(ptr)
        assert got == ("c43a9696fc48bb8a477135a53105e9d2", 410499073)


def test_parse_annex_pointer_skips_binary_content():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        real = td / "real.h5"
        real.write_bytes(b"\x89HDF\r\n\x1a\n" * 1000)
        assert _parse_annex_pointer(real) is None


def test_parse_annex_pointer_skips_garbage_text():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        junk = td / "junk.h5"
        junk.write_text("not an annex pointer at all\n")
        assert _parse_annex_pointer(junk) is None


# --------------------------- format_dataset --------------------------- #

def test_format_dataset_projects_common_schema():
    node = {
        "id": 42,
        "owner": {"login": "USZ_NCH"},
        "name": "Human_MTL",
        "full_name": "USZ_NCH/Human_MTL",
        "description": "Boran et al. 2020",
        "private": False,
        "html_url": "https://gin.g-node.org/USZ_NCH/Human_MTL",
        "clone_url": "https://gin.g-node.org/USZ_NCH/Human_MTL.git",
        "ssh_url": "git@gin.g-node.org:USZ_NCH/Human_MTL.git",
        "stars_count": 7,
        "forks_count": 3,
        "watchers_count": 4,
        "default_branch": "master",
        "size": 5219328,           # KB
        "created_at": "2019-12-03T09:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    out = format_dataset(node)
    assert out["id"] == "USZ_NCH/Human_MTL"
    assert out["owner"] == "USZ_NCH"
    assert out["name"] == "Human_MTL"
    assert out["source"] == "gin"
    assert out["default_branch"] == "master"
    # 5_219_328 KB ≈ 4.98 GB
    assert 4.9 < out["size_gb"] < 5.1
    assert out["stars"] == 7


# --------------------------- gin_search / gin_info (stubbed) --------------------------- #

def test_gin_search_calls_repos_search_endpoint():
    from scitex_dataset.neuroscience.gin import gin_search
    payload = {"data": [{"id": 1}, {"id": 2}]}
    stub = _StubHttpx([_StubResponse(payload)])
    with _swap_httpx(stub):
        out = gin_search(query="ieeg", limit=10, page=1)
    assert out == [{"id": 1}, {"id": 2}]
    args, kwargs = stub.calls[0]
    assert args[0].endswith("/api/v1/repos/search")
    assert kwargs["params"]["q"] == "ieeg"
    assert kwargs["params"]["limit"] == 10


def test_gin_info_calls_owner_repo_endpoint():
    from scitex_dataset.neuroscience.gin import gin_info
    payload = {"full_name": "USZ_NCH/Human_MTL", "default_branch": "master"}
    stub = _StubHttpx([_StubResponse(payload)])
    with _swap_httpx(stub):
        out = gin_info("USZ_NCH/Human_MTL")
    assert out["default_branch"] == "master"
    args, _ = stub.calls[0]
    assert args[0].endswith("/api/v1/repos/USZ_NCH/Human_MTL")


def test_gin_search_returns_empty_on_http_error():
    import httpx as real_httpx
    from scitex_dataset.neuroscience.gin import gin_search

    class _BoomHttpx:
        def get(self, *_a, **_k):
            raise real_httpx.HTTPError("boom")

    with _swap_httpx(_BoomHttpx()):
        assert gin_search(query="x") == []


# --------------------------- dispatcher routing --------------------------- #

def test_download_dataset_routes_unknown_to_value_error():
    import pytest
    from scitex_dataset import download_dataset
    with pytest.raises(ValueError):
        download_dataset("not-a-real-source", "x/y")


def test_download_dataset_routes_catalog_only_to_notimpl():
    import pytest
    from scitex_dataset import download_dataset
    with pytest.raises(NotImplementedError):
        download_dataset("openneuro", "ds000001")


def test_list_sources_now_includes_gin():
    from scitex_dataset import list_sources
    info = list_sources()
    assert "gin" in info["sources"]
    assert info["sources"]["gin"]["domain"] == "neuroscience"
    assert info["sources"]["gin"]["kind"] == "catalog"
