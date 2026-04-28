#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 23:58:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/tests/test_database.py

"""Tests for local SQLite database module."""

from scitex_dataset import database


def test_get_db_path():
    """Test default database path lives under the canonical local-state layout
    (`<scitex_dir>/dataset/runtime/...`) introduced by the 2026-04-28
    local_state migration."""
    path = database.get_db_path()
    s = str(path)
    assert "dataset" in s and "runtime" in s
    assert path.name == "datasets.db"


def test_get_connection_creates_tables(temp_db_path):
    """Test that connection creates required tables."""
    conn = database._get_connection(temp_db_path)

    # Check tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor}

    assert "datasets" in tables
    assert "metadata" in tables
    assert "datasets_fts" in tables

    conn.close()


def test_insert_dataset(temp_db_path):
    """Test inserting a dataset."""
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

    database._insert_dataset(conn, dataset, "openneuro")
    conn.commit()

    # Verify insertion
    cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", ("openneuro:ds001",))
    row = cursor.fetchone()

    assert row is not None
    assert row["name"] == "Test Dataset"
    assert row["n_subjects"] == 25
    assert row["source"] == "openneuro"

    conn.close()


def test_insert_dataset_upsert(temp_db_path):
    """Test that insert replaces existing dataset."""
    conn = database._get_connection(temp_db_path)

    dataset1 = {"id": "ds001", "name": "Original", "n_subjects": 10}
    dataset2 = {"id": "ds001", "name": "Updated", "n_subjects": 20}

    database._insert_dataset(conn, dataset1, "openneuro")
    conn.commit()
    database._insert_dataset(conn, dataset2, "openneuro")
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM datasets")
    assert cursor.fetchone()[0] == 1

    cursor = conn.execute("SELECT name, n_subjects FROM datasets")
    row = cursor.fetchone()
    assert row["name"] == "Updated"
    assert row["n_subjects"] == 20

    conn.close()


def test_search_empty_db(temp_db_path):
    """Test searching empty database."""
    results = database.search(db_path=temp_db_path)
    assert results == []


def test_search_by_source(temp_db_path):
    """Test searching by source."""
    conn = database._get_connection(temp_db_path)

    for i in range(3):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}"},
            "openneuro" if i < 2 else "dandi",
        )
    conn.commit()
    conn.close()

    results = database.search(source="openneuro", db_path=temp_db_path)
    assert len(results) == 2

    results = database.search(source="dandi", db_path=temp_db_path)
    assert len(results) == 1


def test_search_by_modality(temp_db_path):
    """Test searching by modality."""
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

    results = database.search(modality="eeg", db_path=temp_db_path)
    assert len(results) == 2


def test_search_by_subjects(temp_db_path):
    """Test searching by subject count."""
    conn = database._get_connection(temp_db_path)

    for n_sub in [10, 30, 50]:
        database._insert_dataset(
            conn,
            {"id": f"ds{n_sub}", "name": f"Study {n_sub}", "n_subjects": n_sub},
            "openneuro",
        )
    conn.commit()
    conn.close()

    results = database.search(min_subjects=25, db_path=temp_db_path)
    assert len(results) == 2

    results = database.search(max_subjects=35, db_path=temp_db_path)
    assert len(results) == 2

    results = database.search(min_subjects=25, max_subjects=35, db_path=temp_db_path)
    assert len(results) == 1


def test_search_full_text(temp_db_path):
    """Test full-text search."""
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

    results = database.search(query="alzheimer", db_path=temp_db_path)
    assert len(results) == 1
    assert results[0]["id"] == "ds001"


def test_search_limit_offset(temp_db_path):
    """Test pagination with limit and offset."""
    conn = database._get_connection(temp_db_path)

    for i in range(10):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "name": f"Dataset {i}", "downloads": 100 - i},
            "openneuro",
        )
    conn.commit()
    conn.close()

    results = database.search(limit=3, db_path=temp_db_path)
    assert len(results) == 3

    results = database.search(limit=3, offset=3, db_path=temp_db_path)
    assert len(results) == 3


def test_search_order_by(temp_db_path):
    """Test ordering results."""
    conn = database._get_connection(temp_db_path)

    for i in range(3):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}", "downloads": (i + 1) * 100, "n_subjects": 30 - i * 10},
            "openneuro",
        )
    conn.commit()
    conn.close()

    # Order by downloads (default)
    results = database.search(order_by="downloads", db_path=temp_db_path)
    assert results[0]["downloads"] > results[-1]["downloads"]

    # Order by n_subjects
    results = database.search(order_by="n_subjects", db_path=temp_db_path)
    assert results[0]["n_subjects"] > results[-1]["n_subjects"]


def test_get_stats_no_db(temp_db_path):
    """Test stats when database doesn't exist."""
    non_existent = temp_db_path.parent / "nonexistent.db"
    stats = database.get_stats(db_path=non_existent)

    assert stats["exists"] is False
    assert "not built" in stats["message"].lower()


def test_get_stats(temp_db_path):
    """Test getting database statistics."""
    conn = database._get_connection(temp_db_path)

    for i in range(5):
        database._insert_dataset(
            conn,
            {"id": f"ds{i:03d}"},
            "openneuro" if i < 3 else "dandi",
        )
    conn.commit()
    conn.close()

    stats = database.get_stats(db_path=temp_db_path)

    assert stats["exists"] is True
    assert stats["total_datasets"] == 5
    assert stats["by_source"]["openneuro"] == 3
    assert stats["by_source"]["dandi"] == 2
    assert stats["size_mb"] >= 0


def test_clear_db(temp_db_path):
    """Test clearing the database."""
    # Create database
    conn = database._get_connection(temp_db_path)
    conn.close()

    assert temp_db_path.exists()

    # Clear it
    result = database.clear(db_path=temp_db_path)
    assert result is True
    assert not temp_db_path.exists()

    # Clear again (should return False)
    result = database.clear(db_path=temp_db_path)
    assert result is False


# EOF
