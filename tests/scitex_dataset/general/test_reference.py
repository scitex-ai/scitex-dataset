#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/general/test_reference.py

"""Mirror test for ``src/scitex_dataset/general/reference.py``.

Offline tests register a temporary entry pointing at the REAL loopback HTTP
server (``loopback_server`` in ``tests/conftest.py``) — save/restore of the
registry dict, no mocks. The MNIST test hits the real network and only runs
with ``SCITEX_DATASET_RUN_NETWORK=1`` (marker ``network``).
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset import download_dataset, fetch_reference, list_references
from scitex_dataset.general import reference as _ref

RUN_NETWORK = os.environ.get("SCITEX_DATASET_RUN_NETWORK") == "1"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


@contextmanager
def _registered(key, entry):
    saved = dict(_ref.REFERENCES)
    _ref.REFERENCES[key] = entry
    try:
        yield
    finally:
        _ref.REFERENCES.clear()
        _ref.REFERENCES.update(saved)


@pytest.fixture
def loopback_entry(loopback_server, loopback_payload_sha256):
    base, hits = loopback_server
    entry = {
        "name": "Loopback payload",
        "url": f"{base}/payload.bin",
        "filename": "payload.bin",
        "sha256": loopback_payload_sha256,
        "license": "CC0-1.0",
        "citation": "Loopback, T. (2026). A test payload.",
    }
    with _registered("loopback", entry):
        yield entry, hits


def _attempt(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the test asserts on the type
        return exc
    return None


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_list_references_includes_mnist():
    # Arrange
    expected = "mnist"
    # Act
    ids = [e["id"] for e in list_references()]
    # Assert
    assert expected in ids


def test_every_reference_pins_a_lowercase_sha256():
    # Arrange
    pattern = re.compile(r"^[0-9a-f]{64}$")
    # Act
    bad = [k for k, e in _ref.REFERENCES.items() if not pattern.match(e["sha256"])]
    # Assert
    assert bad == []


def test_every_reference_carries_license_and_citation():
    # Arrange
    entries = _ref.REFERENCES
    # Act
    missing = [
        k for k, e in entries.items() if not (e.get("license") and e.get("citation"))
    ]
    # Assert
    assert missing == []


def test_mnist_entry_pins_the_verified_keras_npz():
    # Arrange
    entry = _ref.REFERENCES["mnist"]
    # Act
    pinned = (entry["url"], entry["sha256"], entry["size_bytes"])
    # Assert
    assert pinned == (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        MNIST_SHA256,
        11_490_434,
    )


# ---------------------------------------------------------------------------
# fetch_reference over loopback
# ---------------------------------------------------------------------------


def test_fetch_reference_returns_record_with_entry_license_and_citation(
    tmp_path, loopback_entry
):
    # Arrange
    entry, _ = loopback_entry
    # Act
    rec = fetch_reference("loopback", tmp_path)
    # Assert
    assert (rec.license, rec.citation, rec.sha256) == (
        entry["license"],
        entry["citation"],
        entry["sha256"],
    )


def test_fetch_reference_is_case_insensitive_and_caches(tmp_path, loopback_entry):
    # Arrange
    _, hits = loopback_entry
    fetch_reference("loopback", tmp_path)
    # Act
    rec = fetch_reference("  LoopBack ", tmp_path)
    # Assert
    assert (rec.cached, hits["payload.bin"]) == (True, 1)


def test_fetch_reference_unknown_name_raises_key_error(tmp_path):
    # Arrange
    name = "no-such-reference"
    # Act
    err = _attempt(lambda: fetch_reference(name, tmp_path))
    # Assert
    assert isinstance(err, KeyError)


def test_download_dataset_routes_reference_source(tmp_path, loopback_entry):
    # Arrange
    entry, _ = loopback_entry
    # Act
    rec = download_dataset("reference", "loopback", tmp_path)
    # Assert
    assert Path(rec.path) == (tmp_path / entry["filename"]).resolve()


def test_download_dataset_reference_requires_dest():
    # Arrange
    source = "reference"
    # Act
    err = _attempt(lambda: download_dataset(source, "mnist"))
    # Assert
    assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# Real network (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    not RUN_NETWORK, reason="network: set SCITEX_DATASET_RUN_NETWORK=1 to run"
)
def test_fetch_reference_mnist_downloads_verified_npz_over_network(tmp_path):
    # Arrange
    dest = Path(os.environ.get("SCITEX_DATASET_NETWORK_DEST") or tmp_path)
    # Act
    rec = fetch_reference("mnist", dest)
    # Assert
    assert (rec.sha256, rec.size_bytes, Path(rec.path).name) == (
        MNIST_SHA256,
        11_490_434,
        "mnist.npz",
    )


# EOF
