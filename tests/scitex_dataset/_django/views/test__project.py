#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/test_views_project.py

"""Use in a project: role matrix, storage gate, shared cache, project list.

Through the REST operations (``/api/v1/dataset/``) the Hub mounts generically,
and through the app's own form.
"""

from __future__ import annotations

import json
import time

import _dataset_test_host as host
from conftest import api_get, api_post, get, post

from scitex_dataset.project import load_sources

P = "owner/alpha"
DS = "reference:fixture-set"
FETCH = f"project-datasets/{DS}/fetch"


def _record(client, user, key="r-1", **extra):
    return api_post(
        client,
        "project-datasets",
        {"project": P, "dataset": DS, **extra},
        user=user,
        key=key,
    )


def _fetch(client, user="owner", key="f-1", project=P, **extra):
    return api_post(client, FETCH, {"project": project, **extra}, user=user, key=key)


def _wait_done(client, response, user="owner", project=P):
    job_id = response.json()["job"]["id"]
    for _ in range(200):
        job = api_get(client, f"jobs/{job_id}", user=user, project=project).json()[
            "data"
        ]
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("fetch job did not finish")


# --- role matrix: use in project ---------------------------------------------


def test_anonymous_use_is_401(client, hub, fixture_reference):
    # Arrange
    data = {"project": P, "dataset": DS}
    # Act
    response = api_post(client, "project-datasets", data)
    # Assert
    assert response.status_code == 401


def test_non_member_use_is_404(client, hub, fixture_reference):
    # Arrange
    user = "stranger"
    # Act
    response = _record(client, user)
    # Assert
    assert response.status_code == 404


def test_read_only_member_use_is_403(client, hub, fixture_reference):
    # Arrange
    user = "reader"
    # Act
    response = _record(client, user)
    # Assert
    assert response.status_code == 403


def test_maintainer_records_the_dataset(client, hub, fixture_reference):
    # Arrange
    user = "maint"
    # Act
    response = _record(client, user)
    # Assert
    assert response.status_code == 201


def test_recording_pins_checksums_in_the_manifest(client, hub, fixture_reference):
    # Arrange
    _record(client, "owner", local_dir="data/fixture")
    # Act
    entry = load_sources(hub)["datasets"][0]
    # Assert
    assert entry["files"][0]["sha256"] == fixture_reference["files"][0]["sha256"]


def test_local_folder_must_stay_under_data(client, hub, fixture_reference):
    # Arrange
    user = "owner"
    # Act
    response = _record(client, user, local_dir="../escape")
    # Assert
    assert response.status_code == 403


def test_the_form_records_and_redirects(client, hub, fixture_reference):
    # Arrange
    data = {"key": DS, "local_dir": "data/fixture"}
    # Act
    response = post(client, f"/apps/dataset/use?project={P}", data, "owner")
    # Assert
    assert (response.status_code, len(load_sources(hub)["datasets"])) == (303, 1)


def test_the_form_refuses_a_read_only_member(client, hub, fixture_reference):
    # Arrange
    data = {"key": DS}
    # Act
    response = post(client, f"/apps/dataset/use?project={P}", data, "reader")
    # Assert
    assert response.status_code == 404


# --- fetch: storage gate and shared cache ------------------------------------


def test_fetch_answers_202_with_a_job(client, hub, fixture_reference):
    # Arrange
    user = "owner"
    # Act
    response = _fetch(client, user, local_dir="data/fixture")
    # Assert
    assert (response.status_code, response.json()["job"]["state"]) == (202, "queued")


def test_fetch_links_verified_files_into_the_project(client, hub, fixture_reference):
    # Arrange
    response = _fetch(client, local_dir="data/fixture")
    # Act
    _wait_done(client, response)
    # Assert
    assert (hub / "data" / "fixture" / "payload.bin").is_file()


def test_fetch_job_reports_done(client, hub, fixture_reference):
    # Arrange
    response = _fetch(client, local_dir="data/fixture")
    # Act
    job = _wait_done(client, response)
    # Assert
    assert (job["state"], job["percent"]) == ("done", 100.0)


def test_fetch_records_the_link_mode(client, hub, fixture_reference):
    # Arrange
    response = _fetch(client, local_dir="data/fixture")
    # Act
    _wait_done(client, response)
    # Assert
    assert load_sources(hub)["datasets"][0]["fetch"]["mode"] == "hardlink"


def test_fetch_asks_the_gate_for_the_incoming_bytes(client, hub, fixture_reference):
    # Arrange
    response = _fetch(client, local_dir="data/fixture")
    # Act
    _wait_done(client, response)
    # Assert
    assert host.GATE["asked"] == [(P, [131_072])]


def test_fetch_over_quota_is_413(client, hub, fixture_reference):
    # Arrange
    host.GATE.update(used=0, quota=1_000)
    # Act
    response = _fetch(client)
    # Assert
    assert (response.status_code, response.json()["code"]) == (413, "host.quota")


def test_fetch_over_quota_writes_nothing(client, hub, fixture_reference):
    # Arrange
    host.GATE.update(used=0, quota=1_000)
    # Act
    _fetch(client)
    # Assert
    assert not (hub / ".scitex" / "dataset" / "sources.json").exists()


def test_fetch_with_spend_blocked_is_402(client, hub, fixture_reference):
    # Arrange
    host.GATE.update(blocked=True)
    # Act
    response = _fetch(client)
    # Assert
    assert response.status_code == 402


def test_fetch_without_a_host_gate_is_refused(client, hub, fixture_reference, settings):
    # Arrange
    settings.SCITEX_STORAGE_GATE = None
    # Act
    response = _fetch(client)
    # Assert
    assert response.status_code == 503


def test_read_only_member_cannot_fetch(client, hub, fixture_reference):
    # Arrange
    user = "reader"
    # Act
    response = _fetch(client, user)
    # Assert
    assert response.status_code == 403


def test_a_read_token_cannot_fetch(client, hub, fixture_reference):
    # Arrange
    host.TOKENS["pat-read"] = ("owner", ("dataset:*:read",))
    # Act
    response = api_post(client, FETCH, {"project": P}, token="pat-read", key="t-1")
    # Assert
    assert response.json()["code"] == "api.insufficient_scope"


def test_second_project_reuses_the_shared_cache(
    client, hub, fixture_reference, tmp_path
):
    # Arrange
    beta = tmp_path / "beta"
    (beta / "data").mkdir(parents=True)
    host.PROJECTS["owner/beta"] = {"root": beta, "roles": {"owner": "write"}}
    _wait_done(client, _fetch(client))
    # Act
    _wait_done(
        client, _fetch(client, key="f-2", project="owner/beta"), project="owner/beta"
    )
    # Assert
    assert fixture_reference["hits"].get("payload.bin") == 1


def test_non_member_cannot_read_a_job(client, hub, fixture_reference):
    # Arrange
    response = _fetch(client)
    job_id = response.json()["job"]["id"]
    _wait_done(client, response)
    # Act
    peek = api_get(client, f"jobs/{job_id}", user="stranger", project=P)
    # Assert
    assert peek.status_code == 404


def test_a_replayed_fetch_returns_the_same_job(client, hub, fixture_reference):
    # Arrange
    first = _fetch(client, key="same")
    _wait_done(client, first)
    # Act
    again = _fetch(client, key="same")
    # Assert
    assert again.json()["job"]["id"] == first.json()["job"]["id"]


# --- project dataset list ----------------------------------------------------


def test_non_member_project_page_is_404(client, hub):
    # Arrange
    url = "/apps/dataset/project/"
    # Act
    response = get(client, url, "stranger", project=P)
    # Assert
    assert response.status_code == 404


def test_anonymous_project_page_redirects_to_login(client, hub):
    # Arrange
    url = "/apps/dataset/project/"
    # Act
    response = get(client, url, project=P)
    # Assert
    assert response.status_code == 302


def test_reader_sees_the_project_page(client, hub):
    # Arrange
    url = "/apps/dataset/project/"
    # Act
    response = get(client, url, "reader", project=P)
    # Assert
    assert response.status_code == 200


def test_non_member_project_list_api_is_404(client, hub):
    # Arrange
    path = "project-datasets"
    # Act
    response = api_get(client, path, user="stranger", project=P)
    # Assert
    assert response.status_code == 404


def test_project_outputs_are_not_listed(client, hub):
    # Arrange
    url = "/apps/dataset/project/"
    # Act
    response = get(client, url, "reader", project=P)
    # Assert
    assert b"metrics.json" not in response.content


def test_pinned_public_bytes_on_disk_are_detected(
    client, hub, fixture_reference, loopback_payload
):
    # Arrange
    (hub / "data" / "public").mkdir()
    (hub / "data" / "public" / "payload.bin").write_bytes(loopback_payload)
    # Act
    response = api_get(client, "project-datasets", user="reader", project=P)
    # Assert
    assert [(d["dataset"], d["local_dir"]) for d in response.json()["data"]] == [
        (DS, "data/public")
    ]


def test_legacy_url_registrations_are_migrated(client, hub):
    # Arrange
    legacy = {
        "schema": "scitex-dataset.registry",
        "version": 1,
        "datasets": [
            {
                "kind": "url",
                "location": "https://doi.org/10.5281/zenodo.1",
                "name": "Z",
            },
            {"kind": "path", "location": "data/results", "name": "outputs"},
        ],
    }
    (hub / ".scitex" / "dataset").mkdir(parents=True)
    (hub / ".scitex" / "dataset" / "registry.json").write_text(json.dumps(legacy))
    # Act
    response = get(client, "/apps/dataset/project/", "reader", project=P)
    # Assert
    assert b"Migrated" in response.content


def test_the_legacy_json_endpoint_points_to_the_rest_api(client, hub):
    # Arrange
    url = f"/apps/dataset/api/datasets?project={P}"
    # Act
    response = get(client, url, "reader")
    # Assert
    assert response.status_code == 410


# EOF
