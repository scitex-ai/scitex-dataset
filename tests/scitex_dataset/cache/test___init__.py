#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/cache/test___init__.py

"""The shared dataset cache over the REAL loopback HTTP server (no mocks)."""

from __future__ import annotations

import pytest

from scitex_dataset.cache import CacheBusy, DatasetCache


@pytest.fixture
def card(loopback_server, loopback_payload_sha256, loopback_payload):
    base, hits = loopback_server
    files = [
        {
            "path": "payload.bin",
            "url": f"{base}/payload.bin",
            "sha256": loopback_payload_sha256,
            "size_bytes": len(loopback_payload),
        }
    ]
    return {
        "key": "reference:loop",
        "repository": "reference",
        "accession": "loop",
        "version": "sha256:abc",
        "title": "Loop",
        "files": files,
        "size_bytes": len(loopback_payload),
        "_hits": hits,
    }


def test_an_empty_cache_reports_not_cached(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    # Act
    status = cache.status(card)
    # Assert
    assert status["state"] == "not-cached"


def test_a_fetched_dataset_is_cached(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    # Act
    cache.fetch(card)
    # Assert
    assert cache.status(card)["state"] == "cached"


def test_a_cached_dataset_is_not_downloaded_again(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    cache.fetch(card)
    # Act
    cache.fetch(card)
    # Assert
    assert card["_hits"]["payload.bin"] == 1


def test_fetch_records_every_file_checksum(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    # Act
    entry = cache.fetch(card)
    # Assert
    assert [f["sha256"] for f in entry["files"]] == [card["files"][0]["sha256"]]


def test_progress_reaches_the_total(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    seen = []
    # Act
    cache.fetch(card, progress=lambda done, total, message: seen.append((done, total)))
    # Assert
    assert seen[-1] == (card["size_bytes"], card["size_bytes"])


def test_a_missing_cached_file_reads_as_partial(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    cache.fetch(card)
    (cache.files_dir(card["key"], card["version"]) / "payload.bin").unlink()
    # Act
    status = cache.status(card)
    # Assert
    assert status["state"] == "partial"


def test_eviction_removes_the_least_recently_used_entry(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    cache.fetch(card)
    other = {**card, "key": "reference:loop2"}
    cache.fetch(other)
    cache.touch(other["key"], other["version"])
    # Act
    result = cache.evict(max_bytes=card["size_bytes"])
    # Assert
    assert [e["key"] for e in result["evicted"]] == ["reference:loop"]


def test_without_a_limit_nothing_is_evicted(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    cache.fetch(card)
    # Act
    result = cache.evict()
    # Assert
    assert result["evicted"] == []


def test_a_dry_run_keeps_the_entry(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    cache.fetch(card)
    # Act
    cache.evict(max_bytes=0, dry_run=True)
    # Assert
    assert cache.status(card)["state"] == "cached"


def test_a_concurrent_fetch_of_the_same_entry_is_refused(tmp_path, card):
    # Arrange
    cache = DatasetCache(tmp_path)
    entry_dir = cache.entry_dir(card["key"], card["version"])
    # Act
    with cache._locked(entry_dir):
        try:
            cache.fetch(card)
            raised = None
        except CacheBusy as exc:
            raised = exc
    # Assert
    assert raised is not None


def test_a_catalogue_only_repository_cannot_be_fetched(tmp_path):
    # Arrange
    cache = DatasetCache(tmp_path)
    card = {
        "key": "openneuro:ds1",
        "repository": "openneuro",
        "accession": "ds1",
        "repository_name": "OpenNeuro",
        "files": [],
    }
    # Act
    try:
        cache.fetch(card)
        raised = None
    except NotImplementedError as exc:
        raised = exc
    # Assert
    assert "cannot download" in str(raised)


# EOF
