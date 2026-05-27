#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/general/test_huggingface.py

"""Tests for HuggingFace dataset fetcher.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The
``huggingface_hub`` symbols are imported lazily inside each helper, so
we swap them by injecting a hand-rolled stub module into
``sys.modules['huggingface_hub']`` for the duration of the test.
Environment variables are swapped via ``os.environ[...]`` with a real
save/restore context manager.
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

# `huggingface_hub` is an optional install (extras = ["huggingface"]).
# Skip the swap-based tests cleanly when it isn't available, instead of
# letting `import huggingface_hub` raise ModuleNotFoundError.
pytest.importorskip(
    "huggingface_hub",
    reason="huggingface_hub not installed; install scitex-dataset[huggingface]",
)

from scitex_dataset.general.huggingface import (  # noqa: E402
    _resolve_local_dir,
    _resolve_token,
    dataset_info,
    download_file,
    fetch_dataset,
    search_datasets,
)

# ---------------------------------------------------------------------------
# Test helpers — no unittest.mock
# ---------------------------------------------------------------------------


@contextmanager
def _swap_env(updates: dict, clear: bool = False) -> Iterator[None]:
    """Swap a set of env vars for the duration of the block.

    If ``clear`` is True, every env var listed in ``updates`` AND any
    var that is currently set will be removed first, then ``updates`` is
    applied. Otherwise this only sets the listed keys.
    """
    saved = dict(os.environ)
    try:
        if clear:
            os.environ.clear()
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextmanager
def _swap_path_home(target: Path) -> Iterator[None]:
    """Replace ``Path.home`` with a lambda returning ``target``."""
    saved = Path.home
    Path.home = staticmethod(lambda: target)  # type: ignore[assignment,method-assign]
    try:
        yield
    finally:
        Path.home = saved  # type: ignore[method-assign]


@contextmanager
def _swap_module(name: str, replacement) -> Iterator[None]:
    """Swap ``sys.modules[name]`` with ``replacement`` for the block."""
    saved = sys.modules.get(name)
    sys.modules[name] = replacement
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


class _DatasetInfoStub:
    """Plain-data stand-in for ``huggingface_hub.dataset_info`` return value."""

    def __init__(
        self,
        *,
        id_: str,
        description: str = "",
        downloads: int = 0,
        likes: int = 0,
        private: bool = False,
        gated: bool = False,
        created_at: str = "2026-01-01",
        last_modified: str = "2026-05-01",
    ):
        self.id = id_
        self.description = description
        self.downloads = downloads
        self.likes = likes
        self.private = private
        self.gated = gated
        self.created_at = created_at
        self.last_modified = last_modified


# ---------------------------------------------------------------------------
# _resolve_token tests
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_hf_token_env_var_takes_first_priority(self):
        # Arrange
        updates = {"HF_TOKEN": "token_from_env"}
        # Act
        with _swap_env(updates):
            token = _resolve_token()
        # Assert
        assert token == "token_from_env"

    def test_hf_token_path_env_var_takes_second_priority(self, tmp_path):
        # Arrange
        token_file = tmp_path / "token.txt"
        token_file.write_text("token_from_file")
        updates = {"HF_TOKEN": None, "HF_TOKEN_PATH": str(token_file)}
        # Act
        with _swap_env(updates, clear=True):
            token = _resolve_token()
        # Assert
        assert token == "token_from_file"

    def test_default_secret_location_takes_third_priority(self, tmp_path):
        # Arrange
        secret_file = (
            tmp_path / ".bash.d" / "secrets" / "access_tokens" / "huggingface.txt"
        )
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text("token_from_default")
        # Act
        with _swap_env({}, clear=True), _swap_path_home(tmp_path):
            token = _resolve_token()
        # Assert
        assert token == "token_from_default"

    def test_returns_none_when_no_token_anywhere(self, tmp_path):
        # Arrange
        # tmp_path has no .bash.d/secrets/...
        # Act
        with _swap_env({}, clear=True), _swap_path_home(tmp_path):
            token = _resolve_token()
        # Assert
        assert token is None


# ---------------------------------------------------------------------------
# _resolve_local_dir tests
# ---------------------------------------------------------------------------


class TestResolveLocalDir:
    def test_explicit_local_dir_parameter_is_returned_resolved(self, tmp_path):
        # Arrange
        explicit = str(tmp_path)
        # Act
        result = _resolve_local_dir("user/dataset", local_dir=explicit)
        # Assert
        assert result == tmp_path

    def test_fallback_to_runtime_directory_when_no_spartan_detected(self):
        # Arrange
        repo = "user/dataset"
        # The new canonical path is under runtime/ via the SciTeX resolver.
        expected = (
            Path.home()
            / ".scitex"
            / "dataset"
            / "runtime"
            / "huggingface"
            / "user_dataset"
        )
        # Act
        result = _resolve_local_dir(repo, local_dir=None, spartan_detect=False)
        # Assert
        assert result == expected


# ---------------------------------------------------------------------------
# dataset_info tests
# ---------------------------------------------------------------------------


class TestDatasetInfo:
    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_live_public_dataset_info_returns_id_field(self):
        # Arrange
        repo = "hf-internal-testing/tiny-random-gpt2"
        # Act
        info = dataset_info(repo, repo_type="model")
        # Assert
        assert "id" in info

    def test_dataset_info_returns_repo_id_when_swapped_module_used(self):
        # Arrange
        stub_info = _DatasetInfoStub(id_="user/dataset", downloads=100)
        stub_module = types.ModuleType("huggingface_hub")
        stub_module.dataset_info = lambda repo_id: stub_info  # type: ignore[attr-defined]
        stub_module.model_info = lambda repo_id: stub_info  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            result = dataset_info("user/dataset")
        # Assert
        assert result["id"] == "user/dataset"

    def test_dataset_info_returns_downloads_when_swapped_module_used(self):
        # Arrange
        stub_info = _DatasetInfoStub(id_="user/dataset", downloads=100)
        stub_module = types.ModuleType("huggingface_hub")
        stub_module.dataset_info = lambda repo_id: stub_info  # type: ignore[attr-defined]
        stub_module.model_info = lambda repo_id: stub_info  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            result = dataset_info("user/dataset")
        # Assert
        assert result["downloads"] == 100


# ---------------------------------------------------------------------------
# search_datasets tests
# ---------------------------------------------------------------------------


class TestSearchDatasets:
    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_live_search_returns_non_empty_list_for_biology_query(self):
        # Arrange
        query = "biology"
        # Act
        results = search_datasets(query, limit=5)
        # Assert
        assert len(results) > 0

    def test_search_datasets_returns_one_result_when_stub_returns_one(self):
        # Arrange
        stub_info = _DatasetInfoStub(
            id_="user/biology_dataset", downloads=50, description="A biology dataset"
        )
        stub_module = types.ModuleType("huggingface_hub")
        stub_module.list_datasets = lambda **kw: [stub_info]  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            results = search_datasets("biology", limit=1)
        # Assert
        assert len(results) == 1

    def test_search_datasets_returns_id_from_stub_record(self):
        # Arrange
        stub_info = _DatasetInfoStub(
            id_="user/biology_dataset", downloads=50, description="A biology dataset"
        )
        stub_module = types.ModuleType("huggingface_hub")
        stub_module.list_datasets = lambda **kw: [stub_info]  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            results = search_datasets("biology", limit=1)
        # Assert
        assert results[0]["id"] == "user/biology_dataset"


# ---------------------------------------------------------------------------
# fetch_dataset tests
# ---------------------------------------------------------------------------


class TestFetchDataset:
    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_live_fetch_of_tiny_public_dataset_returns_existing_path(self, tmp_path):
        # Arrange
        repo = "hf-internal-testing/tiny-random-gpt2"
        # Act
        result = fetch_dataset(
            repo,
            local_dir=str(tmp_path),
            repo_type="model",
            max_workers=1,
        )
        # Assert
        assert result.exists()

    def test_fetch_dataset_with_hf_home_override_invokes_swapped_snapshot(
        self, tmp_path
    ):
        # Arrange
        hf_home_path = tmp_path / "hf_cache"
        hf_home_path.mkdir(exist_ok=True)

        calls: list[dict] = []

        def fake_snapshot(**kwargs):
            calls.append(kwargs)
            return str(tmp_path / "result")

        stub_module = types.ModuleType("huggingface_hub")
        stub_module.snapshot_download = fake_snapshot  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            fetch_dataset(
                "user/dataset",
                local_dir=str(tmp_path),
                hf_home_override=str(hf_home_path),
                max_workers=1,
            )
        # Assert
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# download_file tests
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def test_download_file_returns_path_returned_from_swapped_downloader(
        self, tmp_path
    ):
        # Arrange
        mock_file_path = tmp_path / "downloaded_file.txt"
        mock_file_path.write_text("test content")

        def fake_dl(**kwargs):
            del kwargs
            return str(mock_file_path)

        stub_module = types.ModuleType("huggingface_hub")
        stub_module.hf_hub_download = fake_dl  # type: ignore[attr-defined]
        # Act
        with _swap_module("huggingface_hub", stub_module):
            result = download_file(
                "user/dataset",
                "README.md",
                local_dir=str(tmp_path),
            )
        # Assert
        assert Path(result).name == "downloaded_file.txt"


# EOF
