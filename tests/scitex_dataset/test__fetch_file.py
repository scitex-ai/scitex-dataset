#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/test__fetch_file.py

"""Mirror test for ``src/scitex_dataset/_fetch_file.py``.

Drives the PRODUCTION ``fetch_file`` against a REAL loopback HTTP server
(``loopback_server`` fixture in ``tests/conftest.py``). No mocks, no
network beyond 127.0.0.1.
"""

from __future__ import annotations

import json
import urllib.error

from scitex_dataset import FetchError, FetchRecord, fetch_file

WRONG_SHA = "0" * 64


def _attempt(fn):
    """Run ``fn``; return the exception it raised (swallowed), else None."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the test asserts on the type
        return exc
    return None


def test_fetch_file_downloads_and_verifies_new_file(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    rec = fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    # Assert
    assert (rec.cached, rec.sha256, rec.path) == (
        False,
        loopback_payload_sha256,
        str((tmp_path / "payload.bin").resolve()),
    )


def test_fetch_file_record_size_matches_file_on_disk(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    rec = fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    # Assert
    assert rec.size_bytes == (tmp_path / "payload.bin").stat().st_size


def test_fetch_file_record_to_dict_is_json_serialisable_with_metadata(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    rec = fetch_file(
        f"{base}/payload.bin",
        loopback_payload_sha256,
        tmp_path,
        license="CC0",
        citation="Doe 2026",
    )
    # Act
    payload = json.loads(json.dumps(rec.to_dict()))
    # Assert
    assert payload == {
        "url": f"{base}/payload.bin",
        "sha256": loopback_payload_sha256,
        "size_bytes": rec.size_bytes,
        "path": rec.path,
        "retrieved_at": rec.retrieved_at,
        "cached": False,
        "license": "CC0",
        "citation": "Doe 2026",
    }


def test_fetch_file_retrieved_at_is_iso_utc(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    rec = fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    # Assert
    assert rec.retrieved_at.endswith("+00:00")


def test_fetch_file_cache_hit_does_not_download_again(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, hits = loopback_server
    fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    # Act
    rec = fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    # Assert
    assert (rec.cached, hits["payload.bin"]) == (True, 1)


def test_fetch_file_checksum_mismatch_raises_fetch_error(tmp_path, loopback_server):
    # Arrange
    base, _ = loopback_server
    # Act
    err = _attempt(lambda: fetch_file(f"{base}/payload.bin", WRONG_SHA, tmp_path))
    # Assert
    assert isinstance(err, FetchError) and "sha256 mismatch" in str(err)


def test_fetch_file_checksum_mismatch_leaves_no_partial_file(tmp_path, loopback_server):
    # Arrange
    base, _ = loopback_server
    # Act
    _attempt(lambda: fetch_file(f"{base}/payload.bin", WRONG_SHA, tmp_path))
    # Assert
    assert list(tmp_path.iterdir()) == []


def test_fetch_file_truncated_body_raises_fetch_error(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    err = _attempt(
        lambda: fetch_file(f"{base}/truncated.bin", loopback_payload_sha256, tmp_path)
    )
    # Assert
    assert isinstance(err, FetchError) and "truncated" in str(err)


def test_fetch_file_truncated_body_leaves_no_partial_file(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    _attempt(
        lambda: fetch_file(f"{base}/truncated.bin", loopback_payload_sha256, tmp_path)
    )
    # Assert
    assert list(tmp_path.iterdir()) == []


def test_fetch_file_existing_wrong_file_raises_fetch_error(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    (tmp_path / "payload.bin").write_bytes(b"the user's own data")
    # Act
    err = _attempt(
        lambda: fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    )
    # Assert
    assert isinstance(err, FetchError) and "Refusing to overwrite" in str(err)


def test_fetch_file_existing_wrong_file_is_left_untouched_and_not_downloaded(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, hits = loopback_server
    user_file = tmp_path / "payload.bin"
    user_file.write_bytes(b"the user's own data")
    # Act
    _attempt(
        lambda: fetch_file(f"{base}/payload.bin", loopback_payload_sha256, tmp_path)
    )
    # Assert
    assert (user_file.read_bytes(), hits) == (b"the user's own data", {})


def test_fetch_file_replace_true_overwrites_wrong_existing_file(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    (tmp_path / "payload.bin").write_bytes(b"stale")
    # Act
    rec = fetch_file(
        f"{base}/payload.bin", loopback_payload_sha256, tmp_path, replace=True
    )
    # Assert
    assert (rec.cached, rec.sha256) == (False, loopback_payload_sha256)


def test_fetch_file_explicit_filename_is_used(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    # Act
    rec = fetch_file(
        f"{base}/payload.bin", loopback_payload_sha256, tmp_path, "renamed.bin"
    )
    # Assert
    assert rec.path == str((tmp_path / "renamed.bin").resolve())


def test_fetch_file_creates_missing_dest_dir(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    base, _ = loopback_server
    dest = tmp_path / "a" / "b"
    # Act
    fetch_file(f"{base}/payload.bin", loopback_payload_sha256, dest)
    # Assert
    assert (dest / "payload.bin").is_file()


def test_fetch_file_http_404_propagates_http_error(tmp_path, loopback_server):
    # Arrange
    base, _ = loopback_server
    # Act
    err = _attempt(lambda: fetch_file(f"{base}/missing.bin", WRONG_SHA, tmp_path))
    # Assert
    assert isinstance(err, urllib.error.HTTPError)


def test_fetch_file_http_404_leaves_no_file(tmp_path, loopback_server):
    # Arrange
    base, _ = loopback_server
    # Act
    _attempt(lambda: fetch_file(f"{base}/missing.bin", WRONG_SHA, tmp_path))
    # Assert
    assert list(tmp_path.iterdir()) == []


def test_fetch_file_rejects_malformed_sha256(tmp_path):
    # Arrange
    url = "http://127.0.0.1:9/x.bin"
    # Act
    err = _attempt(lambda: fetch_file(url, "abc", tmp_path))
    # Assert
    assert isinstance(err, ValueError) and "64 hex" in str(err)


def test_fetch_file_rejects_filename_with_directory(tmp_path):
    # Arrange
    url = "http://127.0.0.1:9/x.bin"
    # Act
    err = _attempt(lambda: fetch_file(url, WRONG_SHA, tmp_path, "../escape.bin"))
    # Assert
    assert isinstance(err, ValueError) and "bare name" in str(err)


def test_fetch_record_is_frozen():
    # Arrange
    rec = FetchRecord(
        url="u", sha256=WRONG_SHA, size_bytes=0, path="p", retrieved_at="t", cached=True
    )
    # Act
    err = _attempt(lambda: setattr(rec, "cached", False))
    # Assert
    assert isinstance(err, AttributeError)


def test_fetch_error_is_a_runtime_error():
    # Arrange
    cls = FetchError
    # Act
    is_runtime = issubclass(cls, RuntimeError)
    # Assert
    assert is_runtime


# EOF
