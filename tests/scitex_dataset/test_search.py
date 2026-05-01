#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:27:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_search.py

"""Tests for search functionality."""

from scitex_dataset import search_datasets, sort_datasets

SAMPLE_DATASETS = [
    {
        "id": "ds001",
        "name": "Memory Task Study",
        "modalities": ["mri", "eeg"],
        "primary_modality": "mri",
        "n_subjects": 30,
        "tasks": ["memory recall", "rest"],
        "downloads": 100,
        "readme": "A study about memory and cognition.",
    },
    {
        "id": "ds002",
        "name": "Motor Learning",
        "modalities": ["mri"],
        "primary_modality": "mri",
        "n_subjects": 15,
        "tasks": ["finger tapping"],
        "downloads": 50,
        "readme": None,
    },
    {
        "id": "ds003",
        "name": "EEG Resting State",
        "modalities": ["eeg"],
        "primary_modality": "eeg",
        "n_subjects": 50,
        "tasks": ["rest"],
        "downloads": 200,
        "readme": "Resting state EEG recordings.",
    },
]


def test_search_by_modality():
    """Test filtering by modality."""
    results = search_datasets(SAMPLE_DATASETS, modality="eeg")
    assert len(results) == 2
    assert all("eeg" in d["modalities"] for d in results)


def test_search_by_min_subjects():
    """Test filtering by minimum subjects."""
    results = search_datasets(SAMPLE_DATASETS, min_subjects=20)
    assert len(results) == 2
    assert all(d["n_subjects"] >= 20 for d in results)


def test_search_by_task():
    """Test filtering by task name."""
    results = search_datasets(SAMPLE_DATASETS, task_contains="rest")
    assert len(results) == 2


def test_search_by_text_query():
    """Test text search in name and readme."""
    results = search_datasets(SAMPLE_DATASETS, text_query="memory")
    assert len(results) == 1
    assert results[0]["id"] == "ds001"


def test_search_has_readme():
    """Test filtering by readme presence."""
    results = search_datasets(SAMPLE_DATASETS, has_readme=True)
    assert len(results) == 2
    assert all(d["readme"] for d in results)


def test_search_combined_filters():
    """Test combining multiple filters."""
    results = search_datasets(
        SAMPLE_DATASETS,
        modality="mri",
        min_subjects=20,
    )
    assert len(results) == 1
    assert results[0]["id"] == "ds001"


def test_sort_by_downloads():
    """Test sorting by downloads."""
    results = sort_datasets(SAMPLE_DATASETS, by="downloads", descending=True)
    assert results[0]["id"] == "ds003"
    assert results[-1]["id"] == "ds002"


def test_sort_by_subjects_ascending():
    """Test sorting by subjects ascending."""
    results = sort_datasets(SAMPLE_DATASETS, by="n_subjects", descending=False)
    assert results[0]["id"] == "ds002"
    assert results[-1]["id"] == "ds003"


# EOF
