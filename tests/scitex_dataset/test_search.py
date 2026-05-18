#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/test_search.py

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


def test_search_by_modality_eeg_returns_two_rows():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, modality="eeg")
    # Assert
    assert len(results) == 2


def test_search_by_modality_eeg_returns_only_eeg_datasets():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, modality="eeg")
    # Assert
    assert all("eeg" in d["modalities"] for d in results)


def test_search_by_min_subjects_returns_two_rows():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, min_subjects=20)
    # Assert
    assert len(results) == 2


def test_search_by_min_subjects_returns_only_high_count_datasets():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, min_subjects=20)
    # Assert
    assert all(d["n_subjects"] >= 20 for d in results)


def test_search_by_task_contains_rest_returns_two_rows():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, task_contains="rest")
    # Assert
    assert len(results) == 2


def test_search_text_query_memory_returns_one_row():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, text_query="memory")
    # Assert
    assert len(results) == 1


def test_search_text_query_memory_returns_matching_id():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, text_query="memory")
    # Assert
    assert results[0]["id"] == "ds001"


def test_search_has_readme_true_returns_two_rows():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, has_readme=True)
    # Assert
    assert len(results) == 2


def test_search_has_readme_true_returns_only_non_null_readmes():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, has_readme=True)
    # Assert
    assert all(d["readme"] for d in results)


def test_search_combined_modality_and_min_subjects_returns_one_row():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, modality="mri", min_subjects=20)
    # Assert
    assert len(results) == 1


def test_search_combined_modality_and_min_subjects_returns_matching_id():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = search_datasets(datasets, modality="mri", min_subjects=20)
    # Assert
    assert results[0]["id"] == "ds001"


def test_sort_by_downloads_descending_places_top_downloads_first():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = sort_datasets(datasets, by="downloads", descending=True)
    # Assert
    assert results[0]["id"] == "ds003"


def test_sort_by_downloads_descending_places_low_downloads_last():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = sort_datasets(datasets, by="downloads", descending=True)
    # Assert
    assert results[-1]["id"] == "ds002"


def test_sort_by_n_subjects_ascending_places_lowest_first():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = sort_datasets(datasets, by="n_subjects", descending=False)
    # Assert
    assert results[0]["id"] == "ds002"


def test_sort_by_n_subjects_ascending_places_highest_last():
    # Arrange
    datasets = SAMPLE_DATASETS
    # Act
    results = sort_datasets(datasets, by="n_subjects", descending=False)
    # Assert
    assert results[-1]["id"] == "ds003"


# EOF
