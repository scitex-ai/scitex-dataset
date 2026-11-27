#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_rest/test___init__.py

"""The dataset API declaration follows the normalised grammar, by derivation."""

from __future__ import annotations

import pytest

pytest.importorskip("scitex_sdk.app.api_ops")

from scitex_sdk.app.api_ops import operations  # noqa: E402

from scitex_dataset._rest import PLUGIN  # noqa: E402

EXPECTED = [
    ("GET", "/api/v1/dataset/datasets", "dataset:datasets:read"),
    ("GET", "/api/v1/dataset/datasets/{dataset_id}", "dataset:datasets:read"),
    (
        "GET",
        "/api/v1/dataset/cache-entries/{cache_entry_id}",
        "dataset:cache-entries:read",
    ),
    ("GET", "/api/v1/dataset/project-datasets", "dataset:project-datasets:read"),
    ("POST", "/api/v1/dataset/project-datasets", "dataset:project-datasets:write"),
    (
        "POST",
        "/api/v1/dataset/project-datasets/{project_dataset_id}/fetch",
        "dataset:project-datasets:write",
    ),
    ("GET", "/api/v1/dataset/curated-datasets", "dataset:curated-datasets:read"),
    (
        "GET",
        "/api/v1/dataset/curated-datasets/{curated_dataset_id}",
        "dataset:curated-datasets:read",
    ),
    ("GET", "/api/v1/dataset/jobs/{job_id}", "dataset:jobs:read"),
]


def test_every_operation_has_the_derived_url_and_scope():
    # Arrange
    plugin = PLUGIN
    # Act
    derived = [(op.method, op.url, op.scope) for op in operations(plugin)]
    # Assert
    assert derived == EXPECTED


def test_the_plugin_id_is_the_package_slug():
    # Arrange
    plugin = PLUGIN
    # Act
    plugin_id = plugin.id
    # Assert
    assert plugin_id == "dataset"


def test_project_operations_take_an_owner_slug_project():
    # Arrange
    ops = [
        op for op in operations(PLUGIN) if op.resource in ("project-datasets", "jobs")
    ]
    # Act
    scoped = {op.project_scoped for op in ops}
    # Assert
    assert scoped == {True}


def test_fetch_is_an_async_job():
    # Arrange
    fetch = next(op for op in operations(PLUGIN) if op.verb == "fetch")
    # Act
    status = fetch.success_status
    # Assert
    assert status == 202


def test_the_openapi_fragment_validates_as_a_document_piece():
    # Arrange
    plugin = PLUGIN
    # Act
    fragment = plugin.openapi_fragment()
    # Assert
    assert "/datasets/{dataset_id}" in str(fragment["paths"].keys())


def test_every_python_api_function_exists_for_an_operation():
    # Arrange
    import scitex_dataset

    names = [op.mcp_name.removeprefix("dataset_") for op in operations(PLUGIN)]
    # Act
    missing = [n for n in names if not callable(getattr(scitex_dataset, n, None))]
    # Assert
    assert missing == []


# EOF
