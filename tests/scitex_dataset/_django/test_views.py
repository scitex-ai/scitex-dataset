#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/test_views.py

"""The Dataset app's role matrix through a real host: read, write, 404.

Roles follow the Hub project model: members read, maintainers (members with
write permission) write, non-members and unknown projects get the same 404.
"""

from __future__ import annotations

import hashlib

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

BASE = "/apps/dataset/"
Q = "?project=owner/alpha"


def _client(user=None):
    return Client(HTTP_X_TEST_USER=user) if user else Client()


def _upload(user, name="b.csv", data=b"1,2\n", **extra):
    return _client(user).post(
        BASE + "api/upload" + Q, {"file": SimpleUploadedFile(name, data), **extra}
    )


def _register(user, **fields):
    return _client(user).post(BASE + "api/register" + Q, fields)


# -- read --------------------------------------------------------------------


@pytest.mark.parametrize("user", ["owner", "maint", "reader"])
def test_every_member_can_open_the_page(hub, user):
    # Arrange
    client = _client(user)
    # Act
    response = client.get(BASE + Q)
    # Assert
    assert response.status_code == 200


def test_the_page_shows_the_data_file_checksum(hub):
    # Arrange
    expected = hashlib.sha256(b"x,y\n1,2\n").hexdigest()
    # Act
    html = _client("reader").get(BASE + Q).content.decode()
    # Assert
    assert expected in html


@pytest.mark.parametrize("user", ["owner", "maint", "reader"])
def test_every_member_can_list_datasets_as_json(hub, user):
    # Arrange
    client = _client(user)
    # Act
    body = client.get(BASE + "api/datasets" + Q).json()
    # Assert
    assert [f["path"] for f in body["files"]] == ["data/a.csv"]


@pytest.mark.parametrize(
    "user,can_write", [("owner", True), ("maint", True), ("reader", False)]
)
def test_the_listing_says_who_can_write(hub, user, can_write):
    # Arrange
    client = _client(user)
    # Act
    body = client.get(BASE + "api/datasets" + Q).json()
    # Assert
    assert body["project"]["can_write"] is can_write


def test_a_read_only_member_sees_no_add_forms(hub):
    # Arrange
    client = _client("reader")
    # Act
    html = client.get(BASE + Q).content.decode()
    # Assert
    assert "data-dataset-upload" not in html


def test_a_maintainer_sees_the_add_forms(hub):
    # Arrange
    client = _client("maint")
    # Act
    html = client.get(BASE + Q).content.decode()
    # Assert
    assert "data-dataset-upload" in html and "data-dataset-register" in html


# -- non-members: 404 everywhere -----------------------------------------------


@pytest.mark.parametrize("path", ["", "api/datasets"])
def test_a_non_member_gets_404_on_reads(hub, path):
    # Arrange
    client = _client("stranger")
    # Act
    response = client.get(BASE + path + Q)
    # Assert
    assert response.status_code == 404


def test_a_non_member_gets_404_on_upload(hub):
    # Arrange
    user = "stranger"
    # Act
    response = _upload(user)
    # Assert
    assert response.status_code == 404


def test_a_non_member_gets_404_on_register(hub):
    # Arrange
    user = "stranger"
    # Act
    response = _register(user, path="data/a.csv")
    # Assert
    assert response.status_code == 404


def test_an_unknown_project_answers_exactly_like_a_foreign_one(hub):
    # Arrange
    client = _client("reader")
    # Act
    unknown = client.get(BASE + "api/datasets?project=owner/nope")
    # Assert
    assert (unknown.status_code, unknown.json()["error"]) == (404, "project not found")


def test_an_anonymous_api_call_is_401(hub):
    # Arrange
    client = _client()
    # Act
    response = client.get(BASE + "api/datasets" + Q)
    # Assert
    assert response.status_code == 401


# -- writes ------------------------------------------------------------------------


def test_a_read_only_member_cannot_upload(hub):
    # Arrange
    user = "reader"
    # Act
    response = _upload(user)
    # Assert
    assert (response.status_code, (hub / "data" / "b.csv").exists()) == (403, False)


def test_a_read_only_member_cannot_register(hub):
    # Arrange
    user = "reader"
    # Act
    response = _register(user, path="data/a.csv")
    # Assert
    assert (response.status_code, (hub / ".scitex").exists()) == (403, False)


@pytest.mark.parametrize("user", ["owner", "maint"])
def test_a_maintainer_can_upload(hub, user):
    # Arrange
    data = b"1,2\n"
    # Act
    response = _upload(user, data=data)
    # Assert
    assert (response.status_code, (hub / "data" / "b.csv").read_bytes()) == (201, data)


@pytest.mark.parametrize("user", ["owner", "maint"])
def test_a_maintainer_can_register_a_path(hub, user):
    # Arrange
    fields = {"path": "data/a.csv", "name": "A"}
    # Act
    response = _register(user, **fields)
    # Assert
    assert (response.status_code, response.json()["dataset"]["registered_by"]) == (
        201,
        user,
    )


def test_a_maintainer_can_register_a_url(hub):
    # Arrange
    fields = {"url": "https://example.org/eeg.zip"}
    # Act
    response = _register("maint", **fields)
    # Assert
    assert response.json()["dataset"]["kind"] == "url"


def test_register_needs_exactly_one_of_path_or_url(hub):
    # Arrange
    fields = {"path": "data/a.csv", "url": "https://example.org/x"}
    # Act
    response = _register("maint", **fields)
    # Assert
    assert response.status_code == 400


def test_an_upload_outside_data_is_refused(hub):
    # Arrange
    extra = {"subdir": "../.."}
    # Act
    response = _upload("maint", **extra)
    # Assert
    assert response.status_code == 400


def test_the_html_form_redirects_back_to_the_project_page(hub):
    # Arrange
    extra = {"_redirect": "1"}
    # Act
    response = _upload("maint", **extra)
    # Assert
    assert (
        response.status_code,
        response["Location"].startswith(BASE + "?project=owner%2Falpha"),
    ) == (303, True)


def test_a_registered_dataset_is_listed_for_readers(hub):
    # Arrange
    _register("maint", path="data/a.csv", name="A")
    # Act
    body = _client("reader").get(BASE + "api/datasets" + Q).json()
    # Assert
    assert [(d["name"], d["state"]) for d in body["registered"]] == [("A", "unchanged")]


def test_reads_never_change_the_project_tree(hub):
    # Arrange
    before = sorted(p.relative_to(hub) for p in hub.rglob("*"))
    # Act
    _client("reader").get(BASE + Q)
    # Assert
    assert sorted(p.relative_to(hub) for p in hub.rglob("*")) == before


def test_no_selected_project_shows_the_picker_hint(hub):
    # Arrange
    client = _client("reader")
    # Act
    response = client.get(BASE)
    # Assert
    assert "Choose a project" in response.content.decode()


# EOF
