#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/test_views_public.py

"""Search and dataset pages: any logged-in user; a named project needs membership."""

from __future__ import annotations

from conftest import api_get, get, post


def test_anonymous_search_page_redirects_to_login(client, hub):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(client, url, q="payload")
    # Assert
    assert response.status_code == 302


def test_anonymous_search_api_is_401(client, hub):
    # Arrange
    path = "datasets"
    # Act
    response = api_get(client, path, q="payload")
    # Assert
    assert response.status_code == 401


def test_any_logged_in_user_can_search(client, hub, fixture_reference):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(client, url, "stranger", q="payload", repository="reference")
    # Assert
    assert b"Fixture payload set" in response.content


def test_search_api_returns_the_reference_card(client, hub, fixture_reference):
    # Arrange
    path = "datasets"
    # Act
    response = api_get(
        client, path, user="stranger", q="payload", repository="reference"
    )
    # Assert
    assert [c["key"] for c in response.json()["data"]] == ["reference:fixture-set"]


def test_search_card_shows_a_human_readable_size(client, hub, fixture_reference):
    # Arrange
    path = "datasets"
    # Act
    response = api_get(
        client, path, user="stranger", q="payload", repository="reference"
    )
    # Assert
    assert response.json()["data"][0]["size"] == "131.1 kB"


def test_unreachable_index_is_reported_not_hidden(client, hub, fixture_reference):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(client, url, "stranger", q="payload")
    # Assert
    assert b"index is not available" in response.content


def test_unreachable_index_still_finds_reference_files(client, hub, fixture_reference):
    # Arrange
    path = "datasets"
    # Act
    response = api_get(client, path, user="stranger", q="payload")
    # Assert
    assert response.json()["data"][0]["key"] == "reference:fixture-set"


def test_non_member_naming_a_project_on_search_gets_404(client, hub):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(client, url, "stranger", q="x", project="owner/alpha")
    # Assert
    assert response.status_code == 404


def test_unknown_project_is_the_same_404(client, hub):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(client, url, "owner", q="x", project="owner/nope")
    # Assert
    assert response.status_code == 404


def test_member_can_search_within_the_project(client, hub, fixture_reference):
    # Arrange
    url = "/apps/dataset/search/"
    # Act
    response = get(
        client,
        url,
        "reader",
        q="payload",
        repository="reference",
        project="owner/alpha",
    )
    # Assert
    assert response.status_code == 200


def test_detail_shows_how_to_cite(client, hub, fixture_reference):
    # Arrange
    url = "/apps/dataset/d/reference:fixture-set"
    # Act
    response = get(client, url, "stranger")
    # Assert
    assert b"Fixture, T. (2026). A payload." in response.content


def test_detail_shows_a_fetch_snippet(client, hub, fixture_reference):
    # Arrange
    url = "/apps/dataset/d/reference:fixture-set"
    # Act
    response = get(client, url, "stranger")
    # Assert
    assert b"stx.dataset.fetch_reference(&quot;fixture-set&quot;" in response.content


def test_detail_shows_the_licence(client, hub, fixture_reference):
    # Arrange
    path = "datasets/reference:fixture-set"
    # Act
    response = api_get(client, path, user="stranger")
    # Assert
    assert response.json()["data"]["license"] == "CC0-1.0"


def test_unknown_dataset_is_404(client, hub):
    # Arrange
    url = "/apps/dataset/d/reference:no-such-thing"
    # Act
    response = get(client, url, "stranger")
    # Assert
    assert response.status_code == 404


def test_malformed_dataset_key_is_404(client, hub):
    # Arrange
    url = "/apps/dataset/d/not-a-key"
    # Act
    response = get(client, url, "stranger")
    # Assert
    assert response.status_code == 404


def test_removed_upload_endpoint_answers_410(client, hub):
    # Arrange
    url = "/apps/dataset/api/upload?project=owner/alpha"
    # Act
    response = post(client, url, {}, "owner")
    # Assert
    assert response.status_code == 410


# EOF
