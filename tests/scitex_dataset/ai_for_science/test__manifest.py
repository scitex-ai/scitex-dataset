#!/usr/bin/env python3
"""Tests for ai_for_science._manifest — sha256 + YAML emitter + write_manifest."""

import hashlib

import pytest

from scitex_dataset.ai_for_science import _manifest


class TestSha256File:
    def test_sha256_file_returns_64_hex_chars(self, tmp_path):
        # Arrange
        target = tmp_path / "a.bin"
        target.write_bytes(b"hello")
        # Act
        digest = _manifest.sha256_file(target)
        # Assert
        assert len(digest) == 64

    def test_sha256_file_matches_hashlib_sha256_of_same_bytes(self, tmp_path):
        # Arrange
        payload = b"some bytes for hashing"
        target = tmp_path / "b.bin"
        target.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        # Act
        actual = _manifest.sha256_file(target)
        # Assert
        assert actual == expected


class TestManifestEntriesFor:
    def test_manifest_entries_for_returns_one_entry_per_file(self, tmp_path):
        # Arrange
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("bb")
        # Act
        entries = _manifest.manifest_entries_for(
            [tmp_path / "a.txt", tmp_path / "b.txt"], tmp_path
        )
        # Assert
        assert len(entries) == 2

    def test_manifest_entries_for_skips_missing_files(self, tmp_path):
        # Arrange
        (tmp_path / "exists.txt").write_text("x")
        # Act
        entries = _manifest.manifest_entries_for(
            [tmp_path / "exists.txt", tmp_path / "absent.txt"], tmp_path
        )
        # Assert
        assert [e.path for e in entries] == ["exists.txt"]

    def test_manifest_entries_for_sorts_by_path(self, tmp_path):
        # Arrange
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        # Act
        entries = _manifest.manifest_entries_for(
            [tmp_path / "z.txt", tmp_path / "a.txt"], tmp_path
        )
        # Assert
        assert [e.path for e in entries] == ["a.txt", "z.txt"]


class TestYamlEmitter:
    def test_to_yaml_renders_first_scalar_key_first(self):
        # Arrange
        doc = {"id": "test", "version": "v1", "files": []}
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert rendered.splitlines()[0].startswith("id:")

    def test_to_yaml_renders_empty_files_list_inline(self):
        # Arrange
        doc = {"files": []}
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert "files: []" in rendered

    def test_to_yaml_renders_files_entries_as_dash_indented_dicts(self):
        # Arrange
        doc = {
            "files": [
                {"path": "a.txt", "sha256": "abc", "size": 1},
                {"path": "b.txt", "sha256": "def", "size": 2},
            ]
        }
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert "  - path: a.txt" in rendered

    def test_to_yaml_quotes_string_with_colon(self):
        # Arrange
        doc = {"url": "https://example.com"}
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert 'url: "https://example.com"' in rendered

    def test_to_yaml_renders_empty_string_as_quoted_double_quotes(self):
        # Arrange
        doc = {"mask_seed": ""}
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert 'mask_seed: ""' in rendered

    def test_to_yaml_ends_with_a_trailing_newline(self):
        # Arrange
        doc = {"id": "x"}
        # Act
        rendered = _manifest.to_yaml(doc)
        # Assert
        assert rendered.endswith("\n")


class TestWriteManifest:
    def test_write_manifest_returns_yaml_path_in_manifest_dir(self, tmp_path):
        # Arrange
        tracked = tmp_path / "questions.json"
        tracked.write_text("[]")
        manifest_dir = tmp_path / "manifest"
        # Act
        out = _manifest.write_manifest(
            manifest_dir=manifest_dir,
            id="corebench",
            name="CORE-Bench",
            version="v1",
            source_url="https://example.test",
            benchmark="corebench",
            tracked_paths=[tracked],
            tracked_root=tmp_path,
            mask_seed="",
            prepared_at="2026-06-12T00:00:00Z",
        )
        # Assert
        assert out == manifest_dir / "MANIFEST.yaml"

    def test_write_manifest_pinned_prepared_at_is_byte_idempotent(self, tmp_path):
        # Arrange
        tracked = tmp_path / "questions.json"
        tracked.write_text("[]")
        manifest_dir = tmp_path / "manifest"
        kwargs = dict(
            manifest_dir=manifest_dir,
            id="bixbench",
            name="BixBench",
            version="v1",
            source_url="https://example.test",
            benchmark="bixbench",
            tracked_paths=[tracked],
            tracked_root=tmp_path,
            mask_seed="",
            prepared_at="2026-06-12T00:00:00Z",
        )
        first = _manifest.write_manifest(**kwargs).read_bytes()
        # Act
        second = _manifest.write_manifest(**kwargs).read_bytes()
        # Assert
        assert first == second

    def test_write_manifest_records_tracked_file_sha256(self, tmp_path):
        # Arrange
        tracked = tmp_path / "questions.json"
        payload = b'{"q": 1}'
        tracked.write_bytes(payload)
        expected_digest = hashlib.sha256(payload).hexdigest()
        # Act
        out = _manifest.write_manifest(
            manifest_dir=tmp_path / "m",
            id="biomysterybench",
            name="BioMysteryBench",
            version="v1",
            source_url="https://example.test",
            benchmark="biomysterybench",
            tracked_paths=[tracked],
            tracked_root=tmp_path,
            prepared_at="2026-06-12T00:00:00Z",
        )
        # Assert
        assert expected_digest in out.read_text()

    def test_write_manifest_emits_benchmark_yaml_key(self, tmp_path):
        # Arrange
        tracked = tmp_path / "questions.json"
        tracked.write_text("[]")
        # Act
        out = _manifest.write_manifest(
            manifest_dir=tmp_path / "m",
            id="corebench",
            name="CORE-Bench",
            version="v1",
            source_url="https://example.test",
            benchmark="corebench",
            tracked_paths=[tracked],
            tracked_root=tmp_path,
            prepared_at="2026-06-12T00:00:00Z",
        )
        # Assert
        assert "benchmark: corebench" in out.read_text()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
