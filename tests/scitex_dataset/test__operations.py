#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/test__operations.py

"""The Python API of the dataset operations, on real local projects."""

from __future__ import annotations

import pytest

import scitex_dataset
from scitex_dataset._operations import DatasetNotFound
from scitex_dataset.cache import DatasetCache
from scitex_dataset.general import reference as ref


@pytest.fixture
def pinned(loopback_server, loopback_payload_sha256, loopback_payload):
    """A real reference entry served by the loopback server (restored after)."""
    base, hits = loopback_server
    entry = {
        "name": "Ops payload",
        "files": [
            {
                "path": "payload.bin",
                "url": f"{base}/payload.bin",
                "sha256": loopback_payload_sha256,
                "size_bytes": len(loopback_payload),
            }
        ],
        "license": "CC0-1.0",
        "citation": "Ops (2026).",
    }
    saved = dict(ref.REFERENCES)
    ref.REFERENCES["ops-payload"] = entry
    try:
        yield hits
    finally:
        ref.REFERENCES.clear()
        ref.REFERENCES.update(saved)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "proj" / "data").mkdir(parents=True)
    return tmp_path / "proj"


def test_datasets_get_refuses_an_unknown_id():
    # Arrange
    identifier = "reference:none"
    # Act
    try:
        scitex_dataset.datasets_get(identifier)
        raised = False
    except DatasetNotFound:
        raised = True
    # Assert
    assert raised


def test_create_then_list_shows_the_dataset(project, pinned):
    # Arrange
    scitex_dataset.project_datasets_create(project, "reference:ops-payload")
    # Act
    listed = scitex_dataset.project_datasets_list(project)
    # Assert
    assert [d["key"] for d in listed] == ["reference:ops-payload"]


def test_fetch_waits_until_done(project, pinned, tmp_path):
    # Arrange
    cache = DatasetCache(tmp_path / "cache")
    # Act
    job = scitex_dataset.project_datasets_fetch(
        project, "reference:ops-payload", wait=True, cache=cache
    )
    # Assert
    assert job["state"] == "done"


def test_jobs_get_reads_the_finished_job(project, pinned, tmp_path):
    # Arrange
    cache = DatasetCache(tmp_path / "cache")
    job = scitex_dataset.project_datasets_fetch(
        project, "reference:ops-payload", wait=True, cache=cache
    )
    # Act
    again = scitex_dataset.jobs_get(project, job["id"])
    # Assert
    assert again["percent"] == 100.0


def test_cache_entries_get_after_a_fetch_is_cached(project, pinned, tmp_path):
    # Arrange
    cache = DatasetCache(tmp_path / "cache")
    scitex_dataset.project_datasets_fetch(
        project, "reference:ops-payload", wait=True, cache=cache
    )
    # Act
    status = scitex_dataset.cache_entries_get("reference:ops-payload", cache=cache)
    # Assert
    assert status["state"] == "cached"


def test_curated_list_is_empty_without_prepared_benchmarks(tmp_path):
    # Arrange
    root = tmp_path
    # Act
    listed = scitex_dataset.curated_datasets_list(root)
    # Assert
    assert listed == []


# EOF
