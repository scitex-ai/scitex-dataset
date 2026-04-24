#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/src/scitex_dataset/database.py

"""
Local SQLite database for fast dataset searching.

Usage:
    >>> from scitex_dataset import database as db
    >>> db.build()  # Fetch all sources and build database
    >>> results = db.search("alzheimer EEG", min_subjects=20)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scitex_dev.decorators import supports_return_as

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "scitex-dataset" / "datasets.db"

__all__ = [
    "build",
    "update",
    "search",
    "get_stats",
    "get_db_path",
    "clear",
]


def get_db_path() -> Path:
    """Get the database file path."""
    return DEFAULT_DB_PATH


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get database connection, creating tables if needed."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Create tables if not exist
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            name TEXT,
            created TEXT,
            modified TEXT,
            n_subjects INTEGER DEFAULT 0,
            size_gb REAL DEFAULT 0,
            downloads INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            readme TEXT,
            license TEXT,
            doi TEXT,
            url TEXT,
            modalities TEXT,  -- JSON array
            tasks TEXT,       -- JSON array
            primary_modality TEXT,
            data_json TEXT,   -- Full dataset as JSON
            indexed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_source ON datasets(source);
        CREATE INDEX IF NOT EXISTS idx_modalities ON datasets(modalities);
        CREATE INDEX IF NOT EXISTS idx_n_subjects ON datasets(n_subjects);
        CREATE INDEX IF NOT EXISTS idx_downloads ON datasets(downloads);

        CREATE VIRTUAL TABLE IF NOT EXISTS datasets_fts USING fts5(
            id, name, readme, tasks,
            content='datasets',
            content_rowid='rowid'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS datasets_ai AFTER INSERT ON datasets BEGIN
            INSERT INTO datasets_fts(rowid, id, name, readme, tasks)
            VALUES (new.rowid, new.id, new.name, new.readme, new.tasks);
        END;

        CREATE TRIGGER IF NOT EXISTS datasets_ad AFTER DELETE ON datasets BEGIN
            INSERT INTO datasets_fts(datasets_fts, rowid, id, name, readme, tasks)
            VALUES ('delete', old.rowid, old.id, old.name, old.readme, old.tasks);
        END;

        CREATE TRIGGER IF NOT EXISTS datasets_au AFTER UPDATE ON datasets BEGIN
            INSERT INTO datasets_fts(datasets_fts, rowid, id, name, readme, tasks)
            VALUES ('delete', old.rowid, old.id, old.name, old.readme, old.tasks);
            INSERT INTO datasets_fts(rowid, id, name, readme, tasks)
            VALUES (new.rowid, new.id, new.name, new.readme, new.tasks);
        END;
    """)

    return conn


def _insert_dataset(
    conn: sqlite3.Connection,
    dataset: Dict[str, Any],
    source: str,
) -> None:
    """Insert or update a dataset in the database."""
    modalities = json.dumps(dataset.get("modalities", []))
    tasks = json.dumps(dataset.get("tasks", []))

    conn.execute(
        """
        INSERT OR REPLACE INTO datasets (
            id, source, name, created, modified, n_subjects, size_gb,
            downloads, views, readme, license, doi, url,
            modalities, tasks, primary_modality, data_json, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"{source}:{dataset.get('id', '')}",
            source,
            dataset.get("name"),
            dataset.get("created"),
            dataset.get("modified"),
            dataset.get("n_subjects", 0),
            dataset.get("size_gb", 0),
            dataset.get("downloads", 0),
            dataset.get("views", 0),
            dataset.get("readme"),
            dataset.get("license"),
            dataset.get("doi"),
            dataset.get("url"),
            modalities,
            tasks,
            dataset.get("primary_modality"),
            json.dumps(dataset),
            datetime.now().isoformat(),
        ),
    )


@supports_return_as
def build(
    sources: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
    logger=None,
) -> Dict[str, int]:
    """Build the local database from all sources.

    Parameters
    ----------
    sources : list, optional
        Sources to fetch: ["openneuro", "dandi", "physionet"].
        Default: all sources.
    db_path : Path, optional
        Database file path. Default: ~/.cache/scitex-dataset/datasets.db
    logger : optional
        Logger for progress messages.

    Returns
    -------
    dict
        Count of datasets indexed per source.
    """
    if sources is None:
        sources = [
            "openneuro",
            "dandi",
            "physionet",
            "zenodo",
            "figshare",
            "openml",
            "geo",
            "chembl",
            "moleculenet",
            "clinicaltrials",
        ]

    conn = _get_connection(db_path)
    counts = {}

    for source in sources:
        if logger:
            logger.info(f"Fetching from {source}...")

        try:
            if source == "openneuro":
                from .neuroscience.openneuro import fetch_all_datasets, format_dataset
            elif source == "dandi":
                from .neuroscience.dandi import fetch_all_datasets, format_dataset
            elif source == "physionet":
                from .neuroscience.physionet import fetch_all_datasets, format_dataset
            elif source == "zenodo":
                from .general.zenodo import fetch_all_datasets, format_dataset
            elif source == "figshare":
                from .general.figshare import fetch_all_datasets, format_dataset
            elif source == "openml":
                from .general.openml import fetch_all_datasets, format_dataset
            elif source == "geo":
                from .biology.geo import fetch_all_datasets, format_dataset
            elif source == "chembl":
                from .pharmacology.chembl import fetch_all_datasets, format_dataset
            elif source == "moleculenet":
                from .pharmacology.moleculenet import fetch_all_datasets, format_dataset
            elif source == "clinicaltrials":
                from .medical.clinicaltrials import fetch_all_datasets, format_dataset
            else:
                if logger:
                    logger.warning(f"Unknown source: {source}")
                continue

            raw = fetch_all_datasets(logger=logger)
            datasets = [format_dataset(ds) for ds in raw]

            for ds in datasets:
                _insert_dataset(conn, ds, source)

            counts[source] = len(datasets)

            if logger:
                logger.info(f"Indexed {len(datasets)} from {source}")

        except Exception as exc:
            if logger:
                logger.error(f"Error fetching {source}: {exc}")
            counts[source] = 0

    # Update metadata
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("last_build", datetime.now().isoformat()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("total_datasets", str(sum(counts.values()))),
    )

    conn.commit()
    conn.close()

    return counts


def update(
    source: str,
    db_path: Optional[Path] = None,
    logger=None,
) -> int:
    """Update a single source in the database.

    Parameters
    ----------
    source : str
        Source to update: "openneuro", "dandi", or "physionet".
    db_path : Path, optional
        Database file path.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    int
        Number of datasets indexed.
    """
    result = build(sources=[source], db_path=db_path, logger=logger)
    return result.get(source, 0)


@supports_return_as
def search(
    query: Optional[str] = None,
    source: Optional[str] = None,
    modality: Optional[str] = None,
    min_subjects: Optional[int] = None,
    max_subjects: Optional[int] = None,
    min_downloads: Optional[int] = None,
    has_readme: bool = False,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "downloads",
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Search the local database.

    Parameters
    ----------
    query : str, optional
        Full-text search query (searches name, readme, tasks).
    source : str, optional
        Filter by source: "openneuro", "dandi", "physionet".
    modality : str, optional
        Filter by modality (e.g., "mri", "eeg").
    min_subjects : int, optional
        Minimum number of subjects.
    max_subjects : int, optional
        Maximum number of subjects.
    min_downloads : int, optional
        Minimum download count.
    has_readme : bool
        Only include datasets with readme.
    limit : int
        Maximum results (default: 50).
    offset : int
        Skip first N results (for pagination).
    order_by : str
        Order by: downloads, views, n_subjects, size_gb, name.
    db_path : Path, optional
        Database file path.

    Returns
    -------
    list
        List of matching datasets.
    """
    conn = _get_connection(db_path)

    # Build query
    conditions = []
    params = []

    if query:
        # Use FTS for text search
        conditions.append(
            "id IN (SELECT id FROM datasets_fts WHERE datasets_fts MATCH ?)"
        )
        params.append(query)

    if source:
        conditions.append("source = ?")
        params.append(source)

    if modality:
        conditions.append("(modalities LIKE ? OR primary_modality = ?)")
        params.extend([f'%"{modality}"%', modality])

    if min_subjects is not None:
        conditions.append("n_subjects >= ?")
        params.append(min_subjects)

    if max_subjects is not None:
        conditions.append("n_subjects <= ?")
        params.append(max_subjects)

    if min_downloads is not None:
        conditions.append("downloads >= ?")
        params.append(min_downloads)

    if has_readme:
        conditions.append("readme IS NOT NULL AND readme != ''")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Validate order_by
    valid_orders = ["downloads", "views", "n_subjects", "size_gb", "name", "created"]
    if order_by not in valid_orders:
        order_by = "downloads"

    sql = f"""
        SELECT data_json FROM datasets
        WHERE {where_clause}
        ORDER BY {order_by} DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor = conn.execute(sql, params)
    results = [json.loads(row["data_json"]) for row in cursor]

    conn.close()
    return results


@supports_return_as
def get_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get database statistics.

    Returns
    -------
    dict
        Statistics including counts per source, last build time, etc.
    """
    path = db_path or DEFAULT_DB_PATH

    if not path.exists():
        return {"exists": False, "message": "Database not built. Run: db.build()"}

    conn = _get_connection(db_path)

    # Get counts per source
    cursor = conn.execute(
        "SELECT source, COUNT(*) as count FROM datasets GROUP BY source"
    )
    source_counts = {row["source"]: row["count"] for row in cursor}

    # Get metadata
    cursor = conn.execute("SELECT key, value FROM metadata")
    metadata = {row["key"]: row["value"] for row in cursor}

    # Get total
    cursor = conn.execute("SELECT COUNT(*) as total FROM datasets")
    total = cursor.fetchone()["total"]

    conn.close()

    return {
        "exists": True,
        "path": str(path),
        "total_datasets": total,
        "by_source": source_counts,
        "last_build": metadata.get("last_build"),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def clear(db_path: Optional[Path] = None) -> bool:
    """Delete the database file.

    Returns
    -------
    bool
        True if deleted, False if didn't exist.
    """
    path = db_path or DEFAULT_DB_PATH

    if path.exists():
        path.unlink()
        return True
    return False


# EOF
