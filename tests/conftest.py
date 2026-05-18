#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/conftest.py

"""Shared test fixtures for scitex-dataset.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. HTTP
collaborators are swapped at the module namespace using real
save/restore context managers (see ``_swap_httpx_get`` / etc. in the
per-source test modules).

This module also wires module-import-time subprocess coverage. We use
``os.environ[...] = ...`` (force-set), NOT ``setdefault`` — pytest-cov has
already populated ``COVERAGE_FILE`` to a per-test tmp dir by the time
``conftest.py`` is loaded, so ``setdefault`` would silently no-op.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pin coverage's data file at the repo root and point process_startup
# at our pyproject so child interpreters configure themselves correctly.
os.environ["COVERAGE_PROCESS_START"] = str(_PROJECT_ROOT / "pyproject.toml")
os.environ["COVERAGE_FILE"] = str(_PROJECT_ROOT / ".coverage")


def _ensure_subprocess_coverage_shim() -> None:
    """Drop an idempotent ``.pth`` file in site-packages that auto-starts
    coverage in every child Python interpreter via
    ``coverage.process_startup()``.
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = purelib / "_scitex_dataset_subprocess_coverage.pth"
    shim = (
        "import os, coverage\n"
        "if os.environ.get('COVERAGE_PROCESS_START'):\n"
        "    coverage.process_startup()\n"
    )
    try:
        if not pth.exists() or pth.read_text() != shim:
            pth.write_text(shim)
    except OSError:
        # site-packages may be read-only (e.g. system Python); silently
        # skip — local dev venvs are writable and that's where this matters.
        pass


_ensure_subprocess_coverage_shim()


# Sample OpenNeuro dataset
@pytest.fixture
def openneuro_node():
    """Sample OpenNeuro GraphQL node."""
    return {
        "id": "ds000001",
        "name": "Sample MRI Dataset",
        "created": "2020-01-15T10:30:00Z",
        "public": True,
        "publishDate": "2020-02-01T00:00:00Z",
        "analytics": {"views": 500, "downloads": 150},
        "draft": {
            "modified": "2021-03-10T14:20:00Z",
            "readme": "# Sample Dataset\n\nThis is a sample neuroimaging dataset.",
            "description": {
                "Name": "Sample MRI Dataset",
                "BIDSVersion": "1.6.0",
                "License": "CC0",
                "Authors": ["Researcher One", "Researcher Two"],
            },
            "summary": {
                "modalities": ["mri", "eeg"],
                "primaryModality": "mri",
                "subjects": [f"sub-{i:02d}" for i in range(1, 26)],
                "tasks": ["rest", "memory"],
                "size": 5_368_709_120,  # 5 GB
                "totalFiles": 250,
            },
        },
    }


# Sample DANDI dandiset
@pytest.fixture
def dandi_dandiset():
    """Sample DANDI API response."""
    return {
        "identifier": "000001",
        "created": "2021-05-10T08:00:00Z",
        "modified": "2022-01-20T15:30:00Z",
        "contact_person": "researcher@example.com",
        "embargo_status": "OPEN",
        "draft_version": {
            "name": "Sample Electrophysiology Data",
            "version": "draft",
            "status": "Valid",
            "asset_count": 42,
            "size": 10_737_418_240,  # 10 GB
        },
    }


# Sample PhysioNet database
@pytest.fixture
def physionet_database():
    """Sample PhysioNet API response."""
    return {
        "slug": "sample-eeg-db",
        "title": "Sample EEG Database",
        "version": "1.0.0",
        "abstract": "A collection of EEG recordings for epilepsy research.",
        "doi": "10.13026/xxxx-yyyy",
        "license": {"name": "Open Data Commons Attribution License v1.0"},
        "subject_count": 100,
        "record_count": 500,
        "total_size": 21_474_836_480,  # 20 GB
        "publish_date": "2023-06-15",
        "data_access": "open",
    }


# Sample formatted datasets for search tests
@pytest.fixture
def sample_datasets():
    """Sample formatted datasets for testing search and sorting."""
    return [
        {
            "id": "ds001",
            "name": "Alzheimer's EEG Study",
            "modalities": ["eeg"],
            "primary_modality": "eeg",
            "n_subjects": 50,
            "tasks": ["rest", "memory"],
            "downloads": 200,
            "views": 1_000,
            "readme": "A study on Alzheimer's disease using EEG.",
            "size_gb": 5.0,
        },
        {
            "id": "ds002",
            "name": "Motor Control fMRI",
            "modalities": ["mri"],
            "primary_modality": "mri",
            "n_subjects": 30,
            "tasks": ["motor", "rest"],
            "downloads": 150,
            "views": 800,
            "readme": "Functional MRI study of motor control.",
            "size_gb": 10.0,
        },
        {
            "id": "ds003",
            "name": "Sleep Study",
            "modalities": ["eeg", "meg"],
            "primary_modality": "eeg",
            "n_subjects": 20,
            "tasks": ["sleep"],
            "downloads": 75,
            "views": 400,
            "readme": None,
            "size_gb": 3.0,
        },
        {
            "id": "ds004",
            "name": "Epilepsy iEEG",
            "modalities": ["ieeg"],
            "primary_modality": "ieeg",
            "n_subjects": 10,
            "tasks": ["seizure monitoring"],
            "downloads": 500,
            "views": 2_000,
            "readme": "Intracranial EEG recordings from epilepsy patients.",
            "size_gb": 25.0,
        },
    ]


# Temporary database fixture
@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_datasets.db"


# CLI runner
@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()


# EOF
