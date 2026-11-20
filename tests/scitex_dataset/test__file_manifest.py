"""Real loopback HTTP and filesystem coverage for portable pinned manifests."""

import json

import pytest

from scitex_dataset import (
    FetchError,
    create_file_manifest,
    fetch_file,
    replay_file_manifest,
    validate_file_manifest,
)


def prepared(tmp_path, loopback_server, loopback_payload_sha256):
    base, _ = loopback_server
    source = tmp_path / "source"
    record = fetch_file(base + "/payload.bin", loopback_payload_sha256, source)
    manifest = create_file_manifest(
        [
            {
                **record.to_dict(),
                "checksum_basis": "observed fixture hash",
                "subject": "s1",
            }
        ],
        source,
    )
    return manifest, record


def attempt(call):
    try:
        call()
    except Exception as exc:
        return exc
    return None


def test_export_has_relative_destination_and_metadata(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, record = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    # Act
    entry = json.loads(json.dumps(manifest))["files"][0]
    # Assert
    assert (
        entry["path"],
        entry["sha256"],
        entry["checksum_basis"],
        entry["subject"],
    ) == ("payload.bin", record.sha256, "observed fixture hash", "s1")


def test_replay_materializes_under_a_different_root(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, record = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    # Act
    result = replay_file_manifest(manifest, tmp_path / "another-host-root")
    # Assert
    assert (result[0]["path"], result[0]["sha256"], result[0]["cached"]) == (
        str(tmp_path / "another-host-root/payload.bin"),
        record.sha256,
        False,
    )


def test_offline_replay_does_not_use_http(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    replay_file_manifest(manifest, tmp_path / "dest")
    hits = loopback_server[1]["payload.bin"]
    # Act
    result = replay_file_manifest(manifest, tmp_path / "dest", offline=True)
    # Assert
    assert (result[0]["cached"], loopback_server[1]["payload.bin"]) == (True, hits)


def test_missing_offline_file_does_not_create_root(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    dest = tmp_path / "missing"
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, dest, offline=True))
    # Assert
    assert isinstance(error, FileNotFoundError) and not dest.exists()


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "C:/escape", r"..\escape", "", "a//b"]
)
def test_all_invalid_paths_are_rejected_before_download(
    tmp_path, loopback_server, loopback_payload_sha256, path
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"].append({**manifest["files"][0], "path": path})
    hits = loopback_server[1]["payload.bin"]
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, tmp_path / "dest"))
    # Assert
    assert isinstance(error, ValueError) and loopback_server[1]["payload.bin"] == hits


def test_duplicate_destinations_are_rejected(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"] *= 2
    # Act
    error = attempt(lambda: validate_file_manifest(manifest))
    # Assert
    assert isinstance(error, ValueError) and "Duplicate" in str(error)


def test_symlink_escape_is_rejected_before_download(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    dest = tmp_path / "dest"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (dest / "link").symlink_to(outside, target_is_directory=True)
    manifest["files"][0]["path"] = "link/payload.bin"
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, dest))
    # Assert
    assert isinstance(error, ValueError) and not (outside / "payload.bin").exists()


def test_wrong_declared_size_never_installs_new_file(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"][0]["size_bytes"] += 1
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, tmp_path / "dest"))
    # Assert
    assert (
        isinstance(error, FetchError) and not (tmp_path / "dest/payload.bin").exists()
    )


def test_failed_replacement_preserves_original_file(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"][0]["size_bytes"] += 1
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "payload.bin").write_bytes(b"keep this user's file")
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, dest, replace=True))
    # Assert
    assert (
        isinstance(error, FetchError)
        and (dest / "payload.bin").read_bytes() == b"keep this user's file"
    )


def test_export_refuses_records_outside_source_root(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    _, record = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    # Act
    error = attempt(lambda: create_file_manifest([record], tmp_path / "unrelated"))
    # Assert
    assert isinstance(error, ValueError) and "outside" in str(error)


def test_bad_checksum_preserves_destination_absence(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"][0]["sha256"] = "0" * 64
    # Act
    error = attempt(lambda: replay_file_manifest(manifest, tmp_path / "dest"))
    # Assert
    assert (
        isinstance(error, FetchError) and not (tmp_path / "dest/payload.bin").exists()
    )


def test_null_filename_is_rejected_before_io(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"][0]["path"] = "invalid\x00name"
    # Act
    error = attempt(lambda: validate_file_manifest(manifest))
    # Assert
    assert isinstance(error, ValueError)


def test_relative_http_url_is_rejected_before_io(
    tmp_path, loopback_server, loopback_payload_sha256
):
    # Arrange
    manifest, _ = prepared(tmp_path, loopback_server, loopback_payload_sha256)
    manifest["files"][0]["url"] = "https:relative-file"
    # Act
    error = attempt(lambda: validate_file_manifest(manifest))
    # Assert
    assert isinstance(error, ValueError)
