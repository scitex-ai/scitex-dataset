#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_cli/test__reference_verbs.py

"""Mirror test for ``src/scitex_dataset/_cli/_reference_verbs.py``.

``general reference list|fetch`` driven through click's CliRunner; ``fetch``
pulls from the REAL loopback HTTP server (``tests/conftest.py``). No mocks.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from scitex_dataset._cli import main
from scitex_dataset.general import reference as _ref


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
    base, _ = loopback_server
    entry = {
        "name": "Loopback payload",
        "url": f"{base}/payload.bin",
        "filename": "payload.bin",
        "sha256": loopback_payload_sha256,
        "size_bytes": 0,
        "license": "CC0-1.0",
        "citation": "Loopback, T. (2026).",
    }
    with _registered("loopback", entry):
        yield entry


def test_reference_list_json_includes_mnist(cli_runner):
    # Arrange
    args = ["general", "reference", "list", "--json"]
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert "mnist" in [e["id"] for e in json.loads(result.output)]


def test_reference_list_human_output_names_mnist(cli_runner):
    # Arrange
    args = ["general", "reference", "list"]
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert "mnist" in result.output


def test_reference_fetch_json_emits_verified_record(
    cli_runner, tmp_path, loopback_entry
):
    # Arrange
    args = [
        "general",
        "reference",
        "fetch",
        "loopback",
        "--dest",
        str(tmp_path),
        "--json",
    ]
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert json.loads(result.output)["sha256"] == loopback_entry["sha256"]


def test_reference_fetch_second_run_reports_cached(
    cli_runner, tmp_path, loopback_entry
):
    # Arrange
    args = ["general", "reference", "fetch", "loopback", "--dest", str(tmp_path)]
    cli_runner.invoke(main, args)
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert "cached (verified)" in result.output


def test_reference_fetch_wrong_existing_file_exits_nonzero(
    cli_runner, tmp_path, loopback_entry
):
    # Arrange
    (tmp_path / "payload.bin").write_bytes(b"not the payload")
    args = ["general", "reference", "fetch", "loopback", "--dest", str(tmp_path)]
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert result.exit_code == 1


def test_reference_fetch_unknown_name_exits_nonzero(cli_runner, tmp_path):
    # Arrange
    args = ["general", "reference", "fetch", "nope", "--dest", str(tmp_path)]
    # Act
    result = cli_runner.invoke(main, args)
    # Assert
    assert result.exit_code == 1


# EOF
