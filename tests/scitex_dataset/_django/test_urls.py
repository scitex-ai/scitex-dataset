#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/test_rest_tokens.py

"""Personal access tokens on /api/v1/dataset/, and the 1:1 MCP tools."""

from __future__ import annotations

import _dataset_test_host as host
from conftest import api_get


def test_a_read_token_searches_from_a_script(client, hub, fixture_reference):
    # Arrange
    host.TOKENS["pat-read"] = ("stranger", ("dataset:*:read",))
    # Act
    response = api_get(
        client, "datasets", token="pat-read", q="payload", repository="reference"
    )
    # Assert
    assert [c["id"] for c in response.json()["data"]] == ["reference:fixture-set"]


def test_a_bad_token_is_401_not_anonymous(client, hub):
    # Arrange
    path = "datasets"
    # Act
    response = api_get(client, path, token="not-a-token", q="x")
    # Assert
    assert response.status_code == 401


def test_a_token_scoped_to_another_package_cannot_search(client, hub):
    # Arrange
    host.TOKENS["pat-other"] = ("stranger", ("writer:*:read",))
    # Act
    response = api_get(client, "datasets", token="pat-other", q="x")
    # Assert
    assert response.json()["code"] == "api.insufficient_scope"


def test_the_mcp_tool_answers_like_the_rest_endpoint(client, hub, fixture_reference):
    # Arrange
    from scitex_sdk.app.api_mcp import call_tool, tool_specs

    from scitex_dataset._rest import PLUGIN

    spec = next(s for s in tool_specs(PLUGIN) if s.name == "dataset_datasets_list")
    rest = api_get(
        client, "datasets", user="stranger", q="payload", repository="reference"
    ).json()
    # Act
    body = call_tool(spec, {"q": "payload", "repository": "reference"})
    # Assert
    assert body == rest


def test_every_operation_has_one_mcp_tool():
    # Arrange
    from scitex_sdk.app.api_mcp import tool_specs

    from scitex_dataset._rest import PLUGIN

    # Act
    names = sorted(spec.name for spec in tool_specs(PLUGIN))
    # Assert
    assert names == [
        "dataset_cache_entries_get",
        "dataset_curated_datasets_get",
        "dataset_curated_datasets_list",
        "dataset_datasets_get",
        "dataset_datasets_list",
        "dataset_jobs_get",
        "dataset_project_datasets_create",
        "dataset_project_datasets_fetch",
        "dataset_project_datasets_list",
    ]


def test_an_unknown_v1_path_is_a_problem_not_html(client, hub):
    # Arrange
    path = "nothing"
    # Act
    response = api_get(client, path, user="stranger")
    # Assert
    assert response["Content-Type"] == "application/problem+json"


def test_cache_status_reports_not_cached(client, hub, fixture_reference):
    # Arrange
    path = "cache-entries/reference:fixture-set"
    # Act
    response = api_get(client, path, user="stranger")
    # Assert
    assert response.json()["data"]["state"] == "not-cached"


def test_cache_status_never_reveals_a_host_path(client, hub, fixture_reference):
    # Arrange
    path = "cache-entries/reference:fixture-set"
    # Act
    response = api_get(client, path, user="stranger")
    # Assert
    assert "path" not in response.json()["data"]


# EOF
