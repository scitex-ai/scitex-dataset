#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/neuroscience/test_gin.py

"""Tests for the GIN (G-Node Infrastructure) dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``httpx`` module is imported as ``_httpx`` at the call site; we swap
``gin._httpx`` at the module namespace via a real save/restore
context manager. Filesystem fixtures are real temporary directories.

PA-307 compliance: every test fn carries explicit ``# Arrange / # Act /
# Assert`` markers on their own lines in order, and asserts a single
behavioural property each (split-out from the original combined fns).
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

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


# --------------------------- shared fixtures --------------------------- #

@pytest.fixture
def sample_dataset_node():
    # Arrange: GIN API repo payload shape — covers every field the
    # projector reads, including a size in KB that maps to ~5 GB.
    return {
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


# --------------------------- constants + URLs --------------------------- #

def test_constants_gin_host_is_public_https_url():
    # Arrange
    # Act
    host = GIN_HOST
    # Assert
    assert host == "https://gin.g-node.org"


def test_constants_gin_api_v1_is_versioned_under_host():
    # Arrange
    # Act
    api = GIN_API_V1
    # Assert
    assert api == "https://gin.g-node.org/api/v1"


def test_split_repo_id_returns_owner_and_name_for_plain_input():
    # Arrange
    raw = "USZ_NCH/Human_MTL"
    # Act
    got = _split_repo_id(raw)
    # Assert
    assert got == ("USZ_NCH", "Human_MTL")


def test_split_repo_id_strips_surrounding_slashes():
    # Arrange
    raw = "/USZ_NCH/Human_MTL/"
    # Act
    got = _split_repo_id(raw)
    # Assert
    assert got == ("USZ_NCH", "Human_MTL")


def test_split_repo_id_strips_trailing_dot_git_suffix():
    # Arrange
    raw = "USZ_NCH/Human_MTL.git"
    # Act
    got = _split_repo_id(raw)
    # Assert
    assert got == ("USZ_NCH", "Human_MTL")


def test_split_repo_id_rejects_input_without_slash():
    # Arrange
    raw = "no-slash"
    # Act
    # Assert
    with pytest.raises(ValueError):
        _split_repo_id(raw)


def test_split_repo_id_rejects_input_with_extra_segment():
    # Arrange
    raw = "one/two/three"
    # Act
    # Assert
    with pytest.raises(ValueError):
        _split_repo_id(raw)


def test_split_repo_id_rejects_empty_input():
    # Arrange
    raw = ""
    # Act
    # Assert
    with pytest.raises(ValueError):
        _split_repo_id(raw)


def test_raw_url_percent_encodes_path_segments_with_spaces():
    # Arrange
    repo = "USZ_NCH/Human_MTL"
    path = "data_nix/Sub 01.h5"
    # Act
    url = _raw_url(repo, path, branch="master")
    # Assert
    assert url == (
        "https://gin.g-node.org/USZ_NCH/Human_MTL/raw/master/"
        "data_nix/Sub%2001.h5"
    )


# --------------------------- pointer parser --------------------------- #

def test_parse_annex_pointer_returns_hash_and_size_for_md5_backend():
    # Arrange
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ptr = td / "ptr.h5"
        ptr.write_text(
            "/annex/objects/MD5-s410499073--c43a9696fc48bb8a477135a53105e9d2\n"
        )
        # Act
        got = _parse_annex_pointer(ptr)
        # Assert
        assert got == ("c43a9696fc48bb8a477135a53105e9d2", 410499073)


def test_parse_annex_pointer_returns_none_for_binary_content():
    # Arrange
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        real = td / "real.h5"
        real.write_bytes(b"\x89HDF\r\n\x1a\n" * 1000)
        # Act
        got = _parse_annex_pointer(real)
        # Assert
        assert got is None


def test_parse_annex_pointer_returns_none_for_unrecognised_text():
    # Arrange
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        junk = td / "junk.h5"
        junk.write_text("not an annex pointer at all\n")
        # Act
        got = _parse_annex_pointer(junk)
        # Assert
        assert got is None


# --------------------------- format_dataset --------------------------- #

def test_format_dataset_projects_full_name_into_id(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["id"] == "USZ_NCH/Human_MTL"


def test_format_dataset_projects_owner_login(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["owner"] == "USZ_NCH"


def test_format_dataset_projects_repo_name(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["name"] == "Human_MTL"


def test_format_dataset_tags_source_as_gin(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["source"] == "gin"


def test_format_dataset_carries_default_branch(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["default_branch"] == "master"


def test_format_dataset_converts_size_from_kb_to_gb(sample_dataset_node):
    # Arrange: 5_219_328 KB ≈ 4.98 GB.
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert 4.9 < out["size_gb"] < 5.1


def test_format_dataset_carries_stars_count(sample_dataset_node):
    # Arrange
    # Act
    out = format_dataset(sample_dataset_node)
    # Assert
    assert out["stars"] == 7


# --------------------------- gin_search / gin_info (stubbed) --------------------------- #

def test_gin_search_returns_data_payload():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_search

    payload = {"data": [{"id": 1}, {"id": 2}]}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        out = gin_search(query="ieeg", limit=10, page=1)
    # Assert
    assert out == [{"id": 1}, {"id": 2}]


def test_gin_search_calls_repos_search_endpoint():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_search

    stub = _StubHttpx([_StubResponse({"data": []})])
    # Act
    with _swap_httpx(stub):
        gin_search(query="ieeg", limit=10, page=1)
    # Assert
    args, _kwargs = stub.calls[0]
    assert args[0].endswith("/api/v1/repos/search")


def test_gin_search_forwards_query_string_as_q_param():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_search

    stub = _StubHttpx([_StubResponse({"data": []})])
    # Act
    with _swap_httpx(stub):
        gin_search(query="ieeg", limit=10, page=1)
    # Assert
    _args, kwargs = stub.calls[0]
    assert kwargs["params"]["q"] == "ieeg"


def test_gin_search_forwards_limit_param():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_search

    stub = _StubHttpx([_StubResponse({"data": []})])
    # Act
    with _swap_httpx(stub):
        gin_search(query="ieeg", limit=10, page=1)
    # Assert
    _args, kwargs = stub.calls[0]
    assert kwargs["params"]["limit"] == 10


def test_gin_info_returns_default_branch_from_payload():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_info

    payload = {"full_name": "USZ_NCH/Human_MTL", "default_branch": "master"}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        out = gin_info("USZ_NCH/Human_MTL")
    # Assert
    assert out["default_branch"] == "master"


def test_gin_info_calls_owner_repo_endpoint():
    # Arrange
    from scitex_dataset.neuroscience.gin import gin_info

    payload = {"full_name": "USZ_NCH/Human_MTL", "default_branch": "master"}
    stub = _StubHttpx([_StubResponse(payload)])
    # Act
    with _swap_httpx(stub):
        gin_info("USZ_NCH/Human_MTL")
    # Assert
    args, _ = stub.calls[0]
    assert args[0].endswith("/api/v1/repos/USZ_NCH/Human_MTL")


def test_gin_search_returns_empty_on_http_error():
    # Arrange
    import httpx as real_httpx

    from scitex_dataset.neuroscience.gin import gin_search

    class _BoomHttpx:
        def get(self, *_a, **_k):
            raise real_httpx.HTTPError("boom")

    # Act
    with _swap_httpx(_BoomHttpx()):
        out = gin_search(query="x")
    # Assert
    assert out == []


# --------------------------- dispatcher routing --------------------------- #

def test_download_dataset_raises_value_error_for_unknown_source():
    # Arrange
    from scitex_dataset import download_dataset

    # Act
    # Assert
    with pytest.raises(ValueError):
        download_dataset("not-a-real-source", "x/y")


def test_download_dataset_raises_not_implemented_for_catalog_only_source():
    # Arrange
    from scitex_dataset import download_dataset

    # Act
    # Assert
    with pytest.raises(NotImplementedError):
        download_dataset("openneuro", "ds000001")


def test_list_sources_includes_gin_key():
    # Arrange
    from scitex_dataset import list_sources

    # Act
    info = list_sources()
    # Assert
    assert "gin" in info["sources"]


def test_list_sources_marks_gin_domain_as_neuroscience():
    # Arrange
    from scitex_dataset import list_sources

    # Act
    info = list_sources()
    # Assert
    assert info["sources"]["gin"]["domain"] == "neuroscience"


def test_list_sources_marks_gin_kind_as_catalog():
    # Arrange
    from scitex_dataset import list_sources

    # Act
    info = list_sources()
    # Assert
    assert info["sources"]["gin"]["kind"] == "catalog"
