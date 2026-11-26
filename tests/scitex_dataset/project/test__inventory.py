#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/project/test__inventory.py

"""Listing, registering and uploading a project's datasets on a real tree."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scitex_dataset.project import (
    REGISTRY_RELPATH,
    DatasetError,
    list_datasets,
    register_dataset,
    save_upload,
)

CSV = b"x,y\n1,2\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refusal_status(call):
    """The status of the DatasetError ``call`` raises, or None when it succeeds."""
    try:
        call()
    except DatasetError as exc:
        return exc.status
    return None


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "a.csv").write_bytes(CSV)
    (root / "data" / ".hidden").write_bytes(b"secret")
    (root / "scripts").mkdir()
    (root / "scripts" / "load.py").write_text("print('hi')\n")
    return root


# -- listing -----------------------------------------------------------------


def test_lists_data_files_with_size_and_sha256(project):
    # Arrange
    expected = [
        {
            "path": "data/raw/a.csv",
            "size": len(CSV),
            "sha256": _sha(CSV),
            "sha256_note": None,
            "registered": False,
        }
    ]
    # Act
    listing = list_datasets(project, provenance=False)
    # Assert
    assert listing["files"] == expected


def test_listing_skips_hidden_files(project):
    # Arrange
    hidden = "data/.hidden"
    # Act
    paths = [f["path"] for f in list_datasets(project, provenance=False)["files"]]
    # Assert
    assert hidden not in paths


def test_listing_never_exposes_the_absolute_root(project):
    # Arrange
    root_text = str(project)
    # Act
    text = json.dumps(list_datasets(project, provenance=False))
    # Assert
    assert root_text not in text


def test_listing_writes_nothing(project):
    # Arrange
    before = sorted(p.relative_to(project) for p in project.rglob("*"))
    # Act
    list_datasets(project, provenance=False)
    # Assert
    assert sorted(p.relative_to(project) for p in project.rglob("*")) == before


def test_a_symlink_out_of_the_project_is_not_listed(project, tmp_path):
    # Arrange
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"leak")
    (project / "data" / "link.csv").symlink_to(outside)
    # Act
    paths = [f["path"] for f in list_datasets(project, provenance=False)["files"]]
    # Assert
    assert "data/link.csv" not in paths


def test_a_missing_workspace_is_404(tmp_path):
    # Arrange
    absent = tmp_path / "absent"
    # Act
    status = _refusal_status(lambda: list_datasets(absent, provenance=False))
    # Assert
    assert status == 404


# -- registering ---------------------------------------------------------------


def test_register_a_project_path_records_its_checksum(project):
    # Arrange
    location = "data/raw/a.csv"
    # Act
    entry = register_dataset(project, location, name="A", registered_by="alice")
    # Assert
    assert (
        entry["kind"],
        entry["location"],
        entry["sha256"],
        entry["registered_by"],
    ) == ("path", location, _sha(CSV), "alice")


def test_a_registration_is_persisted_in_the_project_registry(project):
    # Arrange
    entry = register_dataset(project, "data/raw/a.csv")
    # Act
    stored = json.loads((project / REGISTRY_RELPATH).read_text())
    # Assert
    assert [d["id"] for d in stored["datasets"]] == [entry["id"]]


def test_an_unchanged_registered_file_is_reported_unchanged(project):
    # Arrange
    register_dataset(project, "data/raw/a.csv")
    # Act
    view = list_datasets(project, provenance=False)
    # Assert
    assert (view["registered"][0]["state"], view["files"][0]["registered"]) == (
        "unchanged",
        True,
    )


def test_a_changed_registered_file_is_reported_changed(project):
    # Arrange
    register_dataset(project, "data/raw/a.csv")
    (project / "data" / "raw" / "a.csv").write_bytes(b"x,y\n9,9\n")
    # Act
    view = list_datasets(project, provenance=False)
    # Assert
    assert view["registered"][0]["state"] == "changed"


def test_register_a_directory_gives_size_and_a_tree_digest(project):
    # Arrange
    location = "data/raw"
    # Act
    entry = register_dataset(project, location)
    # Assert
    assert (entry["size"], len(entry["sha256"])) == (len(CSV), 64)


def test_register_a_url_records_it_without_fetching(project):
    # Arrange
    url = "https://example.org/data/eeg.zip"
    # Act
    entry = register_dataset(project, url, kind="url")
    # Assert
    assert (entry["kind"], entry["name"], entry["size"], entry["sha256"]) == (
        "url",
        "eeg.zip",
        None,
        None,
    )


def test_a_registered_url_is_listed_as_external(project):
    # Arrange
    register_dataset(project, "https://example.org/data/eeg.zip", kind="url")
    # Act
    view = list_datasets(project, provenance=False)
    # Assert
    assert view["registered"][0]["state"] == "external"


@pytest.mark.parametrize("location", ["../outside.csv", "/etc/passwd", "data/../../x"])
def test_paths_outside_the_project_are_refused(project, location):
    # Arrange
    allowed = (403, 404)
    # Act
    status = _refusal_status(lambda: register_dataset(project, location))
    # Assert
    assert status in allowed


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "file:///etc/passwd", "ftp://host/x"]
)
def test_only_http_urls_register(project, url):
    # Arrange
    kind = "url"
    # Act
    call = lambda: register_dataset(project, url, kind=kind)  # noqa: E731
    # Assert
    with pytest.raises(DatasetError):
        call()


def test_a_url_given_as_a_path_is_refused(project):
    # Arrange
    url = "https://example.org/x"
    # Act
    call = lambda: register_dataset(project, url, kind="path")  # noqa: E731
    # Assert
    with pytest.raises(DatasetError):
        call()


def test_a_path_given_as_a_url_is_refused(project):
    # Arrange
    path = "data/raw/a.csv"
    # Act
    call = lambda: register_dataset(project, path, kind="url")  # noqa: E731
    # Assert
    with pytest.raises(DatasetError):
        call()


def test_a_missing_path_is_404(project):
    # Arrange
    location = "data/none.csv"
    # Act
    status = _refusal_status(lambda: register_dataset(project, location))
    # Assert
    assert status == 404


def test_duplicate_registration_is_409(project):
    # Arrange
    register_dataset(project, "data/raw/a.csv")
    # Act
    status = _refusal_status(lambda: register_dataset(project, "data/raw/a.csv"))
    # Assert
    assert status == 409


# -- uploading -------------------------------------------------------------------


def test_upload_writes_into_data_and_reports_size_and_sha256(project):
    # Arrange
    chunks = [b"1,", b"2\n"]
    # Act
    saved = save_upload(project, "b.csv", chunks, subdir="raw")
    # Assert
    assert saved == {"path": "data/raw/b.csv", "size": 4, "sha256": _sha(b"1,2\n")}


def test_upload_bytes_land_on_disk(project):
    # Arrange
    payload = io.BytesIO(b"1,2\n")
    # Act
    save_upload(project, "b.csv", payload)
    # Assert
    assert (project / "data" / "b.csv").read_bytes() == b"1,2\n"


def test_upload_never_overwrites(project):
    # Arrange
    save_upload(project, "b.csv", [b"first"])
    # Act
    status = _refusal_status(lambda: save_upload(project, "b.csv", [b"second"]))
    # Assert
    assert (status, (project / "data" / "b.csv").read_bytes()) == (
        409,
        b"first",
    )


def test_upload_leaves_no_temporary_file(project):
    # Arrange
    save_upload(project, "b.csv", [b"x"])
    # Act
    leftovers = [
        p for p in (project / "data").iterdir() if p.name.startswith(".upload-")
    ]
    # Assert
    assert leftovers == []


@pytest.mark.parametrize("name", ["..", ".env", "", "a\x00b.csv", "-x"])
def test_upload_refuses_unsafe_names(project, name):
    # Arrange
    chunks = [b"x"]
    # Act
    call = lambda: save_upload(project, name, chunks)  # noqa: E731
    # Assert
    with pytest.raises(DatasetError):
        call()


def test_upload_refuses_an_unsafe_subdir(project):
    # Arrange
    subdir = "../scripts"
    # Act
    call = lambda: save_upload(project, "ok.csv", [b"x"], subdir=subdir)  # noqa: E731
    # Assert
    with pytest.raises(DatasetError):
        call()


# EOF
