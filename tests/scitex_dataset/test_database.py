#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-05-18 00:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/scitex_dataset/test_database.py

"""Tests for local SQLite database module."""

from scitex_dataset import database


def test_get_db_path_lives_under_dataset_runtime_layout():
    # Arrange
    expected_substrings = ("dataset", "runtime")
    # Act
    path = database.get_db_path()
    # Assert
    assert all(s in str(path) for s in expected_substrings)


def test_get_db_path_basename_is_datasets_db():
    # Arrange
    expected_name = "datasets.db"
    # Act
    path = database.get_db_path()
    # Assert
    assert path.name == expected_name


def test_get_connection_creates_datasets_table(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    # Act
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor}
    conn.close()
    # Assert
    assert "datasets" in tables


def test_get_connection_creates_metadata_table(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    # Act
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor}
    conn.close()
    # Assert
    assert "metadata" in tables


def test_get_connection_creates_datasets_fts_table(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    # Act
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor}
    conn.close()
    # Assert
    assert "datasets_fts" in tables


def test_insert_dataset_persists_row_under_namespaced_id(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset = {
        "id": "ds001",
        "name": "Test Dataset",
        "n_subjects": 25,
        "size_gb": 5.0,
        "downloads": 100,
        "modalities": ["mri", "eeg"],
        "tasks": ["rest"],
    }
    # Act
    database._insert_dataset(conn, dataset, "openneuro")
    conn.commit()
    cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", ("openneuro:ds001",))
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row is not None


def test_insert_dataset_persists_name(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset = {"id": "ds001", "name": "Test Dataset", "n_subjects": 25}
    # Act
    database._insert_dataset(conn, dataset, "openneuro")
    conn.commit()
    cursor = conn.execute(
        "SELECT name FROM datasets WHERE id = ?", ("openneuro:ds001",)
    )
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row["name"] == "Test Dataset"


def test_insert_dataset_persists_n_subjects(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset = {"id": "ds001", "name": "Test", "n_subjects": 25}
    # Act
    database._insert_dataset(conn, dataset, "openneuro")
    conn.commit()
    cursor = conn.execute(
        "SELECT n_subjects FROM datasets WHERE id = ?", ("openneuro:ds001",)
    )
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row["n_subjects"] == 25


def test_insert_dataset_persists_source(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset = {"id": "ds001", "name": "Test"}
    # Act
    database._insert_dataset(conn, dataset, "openneuro")
    conn.commit()
    cursor = conn.execute(
        "SELECT source FROM datasets WHERE id = ?", ("openneuro:ds001",)
    )
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row["source"] == "openneuro"


def test_insert_dataset_upsert_keeps_single_row(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset1 = {"id": "ds001", "name": "Original", "n_subjects": 10}
    dataset2 = {"id": "ds001", "name": "Updated", "n_subjects": 20}
    # Act
    database._insert_dataset(conn, dataset1, "openneuro")
    conn.commit()
    database._insert_dataset(conn, dataset2, "openneuro")
    conn.commit()
    cursor = conn.execute("SELECT COUNT(*) FROM datasets")
    row_count = cursor.fetchone()[0]
    conn.close()
    # Assert
    assert row_count == 1


def test_insert_dataset_upsert_overwrites_name(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset1 = {"id": "ds001", "name": "Original", "n_subjects": 10}
    dataset2 = {"id": "ds001", "name": "Updated", "n_subjects": 20}
    # Act
    database._insert_dataset(conn, dataset1, "openneuro")
    conn.commit()
    database._insert_dataset(conn, dataset2, "openneuro")
    conn.commit()
    cursor = conn.execute("SELECT name FROM datasets")
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row["name"] == "Updated"


def test_insert_dataset_upsert_overwrites_n_subjects(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    dataset1 = {"id": "ds001", "name": "Original", "n_subjects": 10}
    dataset2 = {"id": "ds001", "name": "Updated", "n_subjects": 20}
    # Act
    database._insert_dataset(conn, dataset1, "openneuro")
    conn.commit()
    database._insert_dataset(conn, dataset2, "openneuro")
    conn.commit()
    cursor = conn.execute("SELECT n_subjects FROM datasets")
    row = cursor.fetchone()
    conn.close()
    # Assert
    assert row["n_subjects"] == 20


def test_search_empty_db_returns_empty_list(temp_db_path):
    # Arrange
    expected = []
    # Act
    results = database.search(db_path=temp_db_path)
    # Assert
    assert results == expected


def test_search_by_source_openneuro_returns_two_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(3):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}"},
            "openneuro" if i < 2 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(source="openneuro", db_path=temp_db_path)
    # Assert
    assert len(results) == 2


def test_search_by_source_dandi_returns_one_row(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(3):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}"},
            "openneuro" if i < 2 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(source="dandi", db_path=temp_db_path)
    # Assert
    assert len(results) == 1


def test_search_by_modality_eeg_returns_two_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    database._insert_dataset(
        conn,
        {"id": "ds001", "name": "MRI Study", "modalities": ["mri"]},
        "openneuro",
    )
    database._insert_dataset(
        conn,
        {"id": "ds002", "name": "EEG Study", "modalities": ["eeg"]},
        "openneuro",
    )
    database._insert_dataset(
        conn,
        {"id": "ds003", "name": "Multimodal", "modalities": ["mri", "eeg"]},
        "openneuro",
    )
    conn.commit()
    conn.close()
    # Act
    results = database.search(modality="eeg", db_path=temp_db_path)
    # Assert
    assert len(results) == 2


def test_search_by_min_subjects_returns_only_high_count_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for n_sub in [10, 30, 50]:
        database._insert_dataset(
            conn,
            {"id": f"ds{n_sub}", "name": f"Study {n_sub}", "n_subjects": n_sub},
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(min_subjects=25, db_path=temp_db_path)
    # Assert
    assert len(results) == 2


def test_search_by_max_subjects_returns_only_low_count_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for n_sub in [10, 30, 50]:
        database._insert_dataset(
            conn,
            {"id": f"ds{n_sub}", "name": f"Study {n_sub}", "n_subjects": n_sub},
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(max_subjects=35, db_path=temp_db_path)
    # Assert
    assert len(results) == 2


def test_search_by_min_and_max_subjects_returns_one_row(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for n_sub in [10, 30, 50]:
        database._insert_dataset(
            conn,
            {"id": f"ds{n_sub}", "name": f"Study {n_sub}", "n_subjects": n_sub},
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(min_subjects=25, max_subjects=35, db_path=temp_db_path)
    # Assert
    assert len(results) == 1


def test_search_full_text_query_returns_one_match(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    database._insert_dataset(
        conn,
        {"id": "ds001", "name": "Alzheimer Study", "readme": "Memory impairment"},
        "openneuro",
    )
    database._insert_dataset(
        conn,
        {"id": "ds002", "name": "Motor Control", "readme": "Movement analysis"},
        "openneuro",
    )
    conn.commit()
    conn.close()
    # Act
    results = database.search(query="alzheimer", db_path=temp_db_path)
    # Assert
    assert len(results) == 1


def test_search_full_text_query_returns_matching_id(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    database._insert_dataset(
        conn,
        {"id": "ds001", "name": "Alzheimer Study", "readme": "Memory impairment"},
        "openneuro",
    )
    database._insert_dataset(
        conn,
        {"id": "ds002", "name": "Motor Control", "readme": "Movement analysis"},
        "openneuro",
    )
    conn.commit()
    conn.close()
    # Act
    results = database.search(query="alzheimer", db_path=temp_db_path)
    # Assert
    assert results[0]["id"] == "ds001"


def test_search_with_limit_returns_at_most_three_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(10):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}", "downloads": 100 - i},
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(limit=3, db_path=temp_db_path)
    # Assert
    assert len(results) == 3


def test_search_with_offset_returns_next_page_of_three_rows(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(10):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}", "downloads": 100 - i},
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(limit=3, offset=3, db_path=temp_db_path)
    # Assert
    assert len(results) == 3


def test_search_order_by_downloads_returns_descending_first_row(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(3):
        database._insert_dataset(
            conn,
            {
                "id": f"ds{i:03d}",
                "downloads": (i + 1) * 100,
                "n_subjects": 30 - i * 10,
            },
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(order_by="downloads", db_path=temp_db_path)
    # Assert
    assert results[0]["downloads"] > results[-1]["downloads"]


def test_search_order_by_n_subjects_returns_descending_first_row(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(3):
        database._insert_dataset(
            conn,
            {
                "id": f"ds{i:03d}",
                "downloads": (i + 1) * 100,
                "n_subjects": 30 - i * 10,
            },
            "openneuro",
        )
    conn.commit()
    conn.close()
    # Act
    results = database.search(order_by="n_subjects", db_path=temp_db_path)
    # Assert
    assert results[0]["n_subjects"] > results[-1]["n_subjects"]


def test_get_stats_returns_exists_false_when_db_missing(temp_db_path):
    # Arrange
    non_existent = temp_db_path.parent / "nonexistent.db"
    # Act
    stats = database.get_stats(db_path=non_existent)
    # Assert
    assert stats["exists"] is False


def test_get_stats_returns_not_built_message_when_db_missing(temp_db_path):
    # Arrange
    non_existent = temp_db_path.parent / "nonexistent.db"
    # Act
    stats = database.get_stats(db_path=non_existent)
    # Assert
    assert "not built" in stats["message"].lower()


def test_get_stats_returns_exists_true_when_db_seeded(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(5):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}"},
            "openneuro" if i < 3 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    stats = database.get_stats(db_path=temp_db_path)
    # Assert
    assert stats["exists"] is True


def test_get_stats_returns_total_count_five_when_five_rows_inserted(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(5):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}"},
            "openneuro" if i < 3 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    stats = database.get_stats(db_path=temp_db_path)
    # Assert
    assert stats["total_datasets"] == 5


def test_get_stats_by_source_openneuro_returns_three_after_seed(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(5):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}"},
            "openneuro" if i < 3 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    stats = database.get_stats(db_path=temp_db_path)
    # Assert
    assert stats["by_source"]["openneuro"] == 3


def test_get_stats_by_source_dandi_returns_two_after_seed(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    for i in range(5):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}"},
            "openneuro" if i < 3 else "dandi",
        )
    conn.commit()
    conn.close()
    # Act
    stats = database.get_stats(db_path=temp_db_path)
    # Assert
    assert stats["by_source"]["dandi"] == 2


def test_clear_db_removes_existing_file(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    conn.close()
    # Act
    database.clear(db_path=temp_db_path)
    # Assert
    assert not temp_db_path.exists()


def test_clear_db_returns_true_when_file_existed(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    conn.close()
    # Act
    result = database.clear(db_path=temp_db_path)
    # Assert
    assert result is True


def test_clear_db_returns_false_when_file_already_absent(temp_db_path):
    # Arrange
    conn = database._get_connection(temp_db_path)
    conn.close()
    database.clear(db_path=temp_db_path)
    # Act
    result = database.clear(db_path=temp_db_path)
    # Assert
    assert result is False


# EOF
