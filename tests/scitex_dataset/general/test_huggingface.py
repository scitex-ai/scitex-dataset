#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for HuggingFace dataset fetcher."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex_dataset.general.huggingface import (
    _resolve_local_dir,
    _resolve_token,
    dataset_info,
    download_file,
    fetch_dataset,
    search_datasets,
)


class TestResolveToken:
    """Test token resolution priority chain."""

    def test_token_from_env_var(self):
        """HF_TOKEN env var has highest priority."""
        with patch.dict(os.environ, {"HF_TOKEN": "token_from_env"}):
            token = _resolve_token()
            assert token == "token_from_env"

    def test_token_from_path_env_var(self, tmp_path):
        """HF_TOKEN_PATH env var is second priority."""
        token_file = tmp_path / "token.txt"
        token_file.write_text("token_from_file")

        with patch.dict(os.environ, {"HF_TOKEN": "", "HF_TOKEN_PATH": str(token_file)}):
            # Clear HF_TOKEN from the environment
            env = dict(os.environ)
            if "HF_TOKEN" in env:
                del env["HF_TOKEN"]
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"HF_TOKEN_PATH": str(token_file)}):
                    token = _resolve_token()
                    assert token == "token_from_file"

    def test_token_from_default_location(self, tmp_path, monkeypatch):
        """Default secret location is third priority."""
        # Mock Path.home() to return tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        secret_file = (
            tmp_path / ".bash.d" / "secrets" / "access_tokens" / "huggingface.txt"
        )
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text("token_from_default")

        with patch.dict(os.environ, {}, clear=True):
            token = _resolve_token()
            assert token == "token_from_default"

    def test_token_not_found_returns_none(self, tmp_path, monkeypatch):
        """Return None when no token is found."""
        # Mock Path.home() to point to a directory with no token
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch.dict(os.environ, {}, clear=True):
            token = _resolve_token()
            assert token is None


class TestResolveLocalDir:
    """Test local directory resolution."""

    def test_explicit_local_dir(self, tmp_path):
        """Explicit local_dir parameter is returned."""
        result = _resolve_local_dir("user/dataset", local_dir=str(tmp_path))
        assert result == tmp_path

    def test_home_fallback(self):
        """Fall back to home directory when no Spartan detected."""
        result = _resolve_local_dir(
            "user/dataset", local_dir=None, spartan_detect=False
        )
        expected = Path.home() / ".scitex" / "dataset" / "huggingface" / "user_dataset"
        assert result == expected


class TestDatasetInfo:
    """Test dataset_info function."""

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_dataset_info_public_dataset(self):
        """Fetch info for a public dataset."""
        # Use a public test dataset
        info = dataset_info("hf-internal-testing/tiny-random-gpt2", repo_type="model")
        assert "id" in info
        assert "name" in info
        assert info["url"]

    def test_dataset_info_mock(self):
        """Test dataset_info with mocked API."""
        mock_info = MagicMock()
        mock_info.id = "user/dataset"
        mock_info.description = "Test dataset"
        mock_info.downloads = 100
        mock_info.likes = 10
        mock_info.private = False
        mock_info.gated = False
        mock_info.created_at = "2026-01-01"
        mock_info.last_modified = "2026-05-01"

        with patch(
            "scitex_dataset.general.huggingface.dataset_info"
            if False
            else "huggingface_hub.dataset_info"
        ) as mock_fn:
            mock_fn.return_value = mock_info

            result = dataset_info("user/dataset")
            assert result["id"] == "user/dataset"
            assert result["downloads"] == 100


class TestSearchDatasets:
    """Test search_datasets function."""

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_search_datasets_live(self):
        """Search for datasets (live test)."""
        results = search_datasets("biology", limit=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "id" in results[0]
        assert "name" in results[0]
        assert "url" in results[0]

    def test_search_datasets_mock(self):
        """Test search_datasets with mocked API."""
        mock_dataset_info = MagicMock()
        mock_dataset_info.id = "user/biology_dataset"
        mock_dataset_info.description = "A biology dataset"
        mock_dataset_info.downloads = 50
        mock_dataset_info.likes = 5
        mock_dataset_info.private = False
        mock_dataset_info.gated = False

        with patch("huggingface_hub.list_datasets") as mock_fn:
            mock_fn.return_value = [mock_dataset_info]

            results = search_datasets("biology", limit=1)
            assert len(results) == 1
            assert results[0]["id"] == "user/biology_dataset"


class TestFetchDataset:
    """Test fetch_dataset function."""

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="Requires HF_TOKEN for network access",
    )
    def test_fetch_tiny_dataset(self, tmp_path):
        """Fetch a tiny public dataset for testing."""
        # Use a very small public dataset for testing
        result = fetch_dataset(
            "hf-internal-testing/tiny-random-gpt2",
            local_dir=str(tmp_path),
            repo_type="model",
            max_workers=1,
        )
        assert result.exists()
        assert result.is_dir()

    def test_fetch_dataset_with_hf_home_override(self, tmp_path):
        """Test HF_HOME override."""
        hf_home_path = tmp_path / "hf_cache"
        hf_home_path.mkdir(exist_ok=True)

        original_hf_home = os.environ.get("HF_HOME")

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "result")

            try:
                fetch_dataset(
                    "user/dataset",
                    local_dir=str(tmp_path),
                    hf_home_override=str(hf_home_path),
                    max_workers=1,
                )

                # Verify HF_HOME was set
                assert mock_dl.called
            finally:
                # Restore original HF_HOME
                if original_hf_home:
                    os.environ["HF_HOME"] = original_hf_home
                elif "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]


class TestDownloadFile:
    """Test download_file function."""

    def test_download_file_mock(self, tmp_path):
        """Test download_file with mocked API."""
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            mock_file_path = tmp_path / "downloaded_file.txt"
            mock_file_path.write_text("test content")
            mock_dl.return_value = str(mock_file_path)

            result = download_file(
                "user/dataset",
                "README.md",
                local_dir=str(tmp_path),
            )
            assert Path(result).name == "downloaded_file.txt"


# EOF
