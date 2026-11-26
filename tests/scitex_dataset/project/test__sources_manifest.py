#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/project/test__sources.py

"""The project manifest of public datasets, detection, and migration."""

from __future__ import annotations

import json

import pytest

from scitex_dataset.project import (
    DatasetError,
    detect_datasets,
    load_sources,
    project_datasets,
    record_dataset,
)

CARD = {
    "key": "reference:demo",
    "repository": "reference",
    "accession": "demo",
    "title": "Demo",
    "version": "v1",
    "doi": "10.1/demo",
    "license": "CC0-1.0",
    "citation": "Demo (2026).",
    "files": [
        {"path": "a.bin", "sha256": "0" * 64, "size_bytes": 3, "url": "https://x/a.bin"}
    ],
}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "data").mkdir()
    return tmp_path


def test_reading_a_project_without_a_manifest_writes_nothing(project):
    # Arrange
    root = project
    # Act
    load_sources(root)
    # Assert
    assert not (root / ".scitex").exists()


def test_recording_writes_the_licence(project):
    # Arrange
    root = project
    # Act
    record_dataset(root, CARD, added_by="owner")
    # Assert
    assert load_sources(root)["datasets"][0]["license"] == "CC0-1.0"


def test_recording_twice_keeps_one_entry(project):
    # Arrange
    record_dataset(project, CARD)
    # Act
    record_dataset(project, CARD)
    # Assert
    assert len(load_sources(project)["datasets"]) == 1


def test_the_default_folder_is_under_data(project):
    # Arrange
    root = project
    # Act
    entry = record_dataset(root, CARD)
    # Assert
    assert entry["local_dir"] == "data/demo"


def test_a_folder_outside_data_is_refused(project):
    # Arrange
    root = project
    # Act
    try:
        record_dataset(root, CARD, local_dir="scripts")
        raised = None
    except DatasetError as exc:
        raised = exc
    # Assert
    assert raised.status == 400


def test_a_corrupt_manifest_is_reported(project):
    # Arrange
    (project / ".scitex" / "dataset").mkdir(parents=True)
    (project / ".scitex" / "dataset" / "sources.json").write_text("{nope")
    # Act
    listing = project_datasets(project, with_readers=False)
    # Assert
    assert "not valid JSON" in listing["manifest_error"]


def test_migrated_url_registrations_are_written_on_the_next_write(project):
    # Arrange
    registry = {
        "schema": "scitex-dataset.registry",
        "version": 1,
        "datasets": [
            {"kind": "url", "location": "https://doi.org/10.1/z", "name": "Z"}
        ],
    }
    (project / ".scitex" / "dataset").mkdir(parents=True)
    (project / ".scitex" / "dataset" / "registry.json").write_text(json.dumps(registry))
    # Act
    record_dataset(project, CARD)
    # Assert
    written = json.loads((project / ".scitex" / "dataset" / "sources.json").read_text())
    assert sorted(d["origin"] for d in written["datasets"]) == [
        "app",
        "migrated:registry.json",
    ]


def test_mnist_idx_bytes_on_disk_are_not_detected_from_size_alone(project):
    # Arrange: right size, wrong bytes
    (project / "data" / "mnist").mkdir()
    (project / "data" / "mnist" / "t10k-labels-idx1-ubyte.gz").write_bytes(b"x" * 4_542)
    # Act
    detected = detect_datasets(project)
    # Assert
    assert detected == []


def test_a_listing_never_shows_project_outputs(project):
    # Arrange
    (project / "data" / "results").mkdir()
    (project / "data" / "results" / "metrics.json").write_text("{}")
    # Act
    listing = project_datasets(project, with_readers=False)
    # Assert
    assert listing["datasets"] == []


def test_a_recorded_dataset_without_local_files_reads_missing(project):
    # Arrange
    record_dataset(project, CARD)
    # Act
    listing = project_datasets(project, with_readers=False)
    # Assert
    assert listing["datasets"][0]["local"]["state"] == "missing"


# EOF
