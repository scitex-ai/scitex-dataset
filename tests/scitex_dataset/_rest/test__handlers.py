#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_rest/test__handlers.py

"""The handlers through the SDK executor, as a local MCP caller (no web host)."""

from __future__ import annotations

import pytest

pytest.importorskip("scitex_sdk.app.api_mcp")

from scitex_sdk.app.api_exec import ApiCaller, execute  # noqa: E402
from scitex_sdk.app.api_mcp import call_tool, local_caller, tool_specs  # noqa: E402
from scitex_sdk.app.api_ops import operations  # noqa: E402

from scitex_dataset._rest import PLUGIN  # noqa: E402

OPS = {op.mcp_name: op for op in operations(PLUGIN)}
SPECS = {spec.name: spec for spec in tool_specs(PLUGIN)}


def test_search_finds_the_mnist_idx_reference():
    # Arrange
    arguments = {"q": "mnist idx", "repository": "reference"}
    # Act
    body = call_tool(SPECS["dataset_datasets_list"], arguments)
    # Assert
    assert [d["id"] for d in body["data"]] == ["reference:mnist-idx"]


def test_a_dataset_card_carries_its_licence():
    # Arrange
    arguments = {"dataset_id": "reference:mnist-idx"}
    # Act
    body = call_tool(SPECS["dataset_datasets_get"], arguments)
    # Assert
    assert body["data"]["license"].startswith("CC BY-SA 3.0")


def test_an_unknown_dataset_is_a_404_problem():
    # Arrange
    op = OPS["dataset_datasets_get"]
    # Act
    result = execute(op, local_caller(), {}, item_id="reference:nope")
    # Assert
    assert (result.status, result.body["code"]) == (404, "dataset.not_found")


def test_a_project_records_a_dataset_locally(tmp_path):
    # Arrange
    (tmp_path / "data").mkdir()
    arguments = {"project": str(tmp_path), "dataset": "reference:mnist-idx"}
    # Act
    body = call_tool(SPECS["dataset_project_datasets_create"], arguments)
    # Assert
    assert body["data"]["local_dir"] == "data/mnist-idx"


def test_a_read_only_caller_cannot_record(tmp_path):
    # Arrange
    op = OPS["dataset_project_datasets_create"]
    caller = ApiCaller(user=None, scopes=("dataset:*:read",), via="local")
    # Act
    result = execute(
        op, caller, {"project": str(tmp_path), "dataset": "reference:mnist-idx"}
    )
    # Assert
    assert result.body["code"] == "api.insufficient_scope"


def test_the_project_list_is_empty_for_a_new_project(tmp_path):
    # Arrange
    (tmp_path / "data").mkdir()
    # Act
    body = call_tool(SPECS["dataset_project_datasets_list"], {"project": str(tmp_path)})
    # Assert
    assert body["data"] == []


@pytest.fixture
def empty_dataset_root(tmp_path):
    """``SCITEX_DATASET_ROOT`` pointed at an empty directory, restored after."""
    import os

    previous = os.environ.get("SCITEX_DATASET_ROOT")
    os.environ["SCITEX_DATASET_ROOT"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if previous is None:
            os.environ.pop("SCITEX_DATASET_ROOT", None)
        else:
            os.environ["SCITEX_DATASET_ROOT"] = previous


def test_a_curated_page_for_an_unprepared_benchmark_is_404(empty_dataset_root):
    # Arrange
    op = OPS["dataset_curated_datasets_get"]
    # Act
    result = execute(op, local_caller(), {}, item_id="corebench")
    # Assert
    assert result.status == 404


# EOF
