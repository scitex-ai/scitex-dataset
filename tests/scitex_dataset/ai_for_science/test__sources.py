#!/usr/bin/env python3
"""Tests for ai_for_science._sources (per-capsule source registration).

No mocks / monkeypatch: every check uses real relpaths and real tmp
trees on disk. Covers the doc denylist, the source/doc split, the
filesystem snapshot, and the ``sources.jsonl`` writer (incl. idempotence).
"""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science._sources import (
    classify_capsule_sources,
    is_doc_source,
    snapshot_relpaths,
    write_sources,
)


# ---------------------------------------------------------------------------
# is_doc_source — the documented doc denylist
# ---------------------------------------------------------------------------


class TestIsDocSource:
    @pytest.mark.parametrize(
        "relpath",
        [
            "README.md",
            "readme.txt",
            "code/README",
            "REPRODUCING.md",
            "reproduce.sh",
            "paper.pdf",
            "sub/supplementary.pdf",
            "manuscript_v2.txt",
        ],
    )
    def test_doc_basename_is_flagged_as_doc(self, relpath):
        # Arrange
        # Act
        result = is_doc_source(relpath)
        # Assert
        assert result is True

    @pytest.mark.parametrize(
        "relpath",
        [
            "data/raw.csv",
            "code/analysis.py",
            "results/output.csv",
            "config.yaml",
            "notes.md",
        ],
    )
    def test_source_basename_is_not_flagged_as_doc(self, relpath):
        # Arrange
        # Act
        result = is_doc_source(relpath)
        # Assert
        assert result is False


# ---------------------------------------------------------------------------
# classify_capsule_sources — the source/doc split
# ---------------------------------------------------------------------------


class TestClassifyCapsuleSources:
    def _mixed(self):
        return [
            "results/output.csv",
            "README.md",
            "data/raw.csv",
            "REPRODUCING.md",
            "code/analysis.py",
            "paper.pdf",
        ]

    def test_classify_returns_sorted_sources_only(self):
        # Arrange
        relpaths = self._mixed()
        # Act
        out = classify_capsule_sources(relpaths)
        # Assert
        assert out["sources"] == [
            "code/analysis.py",
            "data/raw.csv",
            "results/output.csv",
        ]

    def test_classify_returns_sorted_excluded_docs_only(self):
        # Arrange
        relpaths = self._mixed()
        # Act
        out = classify_capsule_sources(relpaths)
        # Assert
        assert out["excluded_docs"] == [
            "README.md",
            "REPRODUCING.md",
            "paper.pdf",
        ]

    def test_classify_empty_input_returns_empty_split(self):
        # Arrange
        relpaths = []
        # Act
        out = classify_capsule_sources(relpaths)
        # Assert
        assert out == {"sources": [], "excluded_docs": []}


# ---------------------------------------------------------------------------
# snapshot_relpaths — the filesystem walk
# ---------------------------------------------------------------------------


class TestSnapshotRelpaths:
    def test_snapshot_returns_sorted_file_relpaths_only(self, tmp_path):
        # Arrange
        (tmp_path / "data").mkdir()
        (tmp_path / "code").mkdir()
        (tmp_path / "empty").mkdir()  # empty subdir must be omitted
        (tmp_path / "data" / "raw.csv").write_text("x", encoding="utf-8")
        (tmp_path / "code" / "main.py").write_text("y", encoding="utf-8")
        (tmp_path / "top.txt").write_text("z", encoding="utf-8")
        # Act
        out = snapshot_relpaths(tmp_path)
        # Assert
        assert out == ["code/main.py", "data/raw.csv", "top.txt"]


# ---------------------------------------------------------------------------
# write_sources — the eval/sources.jsonl writer
# ---------------------------------------------------------------------------


def _capsule_rows():
    return [
        {
            "friendly_id": "capsule-000",
            "native_id": "abc",
            "benchmark": "corebench",
            "sources": ["data/raw.csv"],
            "excluded_docs": ["README.md"],
        },
        {
            "friendly_id": "capsule-001",
            "native_id": "def",
            "benchmark": "corebench",
            "sources": ["code/main.py"],
            "excluded_docs": [],
        },
    ]


class TestWriteSources:
    def test_write_sources_returns_sources_path(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "eval"
        # Act
        ret = write_sources(eval_dir=eval_dir, capsule_sources=_capsule_rows())
        # Assert
        assert Path(ret["sources"]) == eval_dir / "sources.jsonl"

    def test_write_sources_returns_capsule_count(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "eval"
        # Act
        ret = write_sources(eval_dir=eval_dir, capsule_sources=_capsule_rows())
        # Assert
        assert ret["n_capsules"] == 2

    def test_write_sources_writes_one_line_per_capsule(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "eval"
        # Act
        write_sources(eval_dir=eval_dir, capsule_sources=_capsule_rows())
        # Assert
        lines = (eval_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["friendly_id"] for line in lines] == [
            "capsule-000",
            "capsule-001",
        ]

    def test_write_sources_creates_missing_eval_dir(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "nested" / "eval"
        # Act
        write_sources(eval_dir=eval_dir, capsule_sources=_capsule_rows())
        # Assert
        assert (eval_dir / "sources.jsonl").is_file()

    def test_write_sources_is_idempotent_on_rewrite(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "eval"
        rows = _capsule_rows()
        write_sources(eval_dir=eval_dir, capsule_sources=rows)
        first = (eval_dir / "sources.jsonl").read_text(encoding="utf-8")
        # Act
        write_sources(eval_dir=eval_dir, capsule_sources=rows)
        second = (eval_dir / "sources.jsonl").read_text(encoding="utf-8")
        # Assert
        assert first == second

    def test_write_sources_empty_list_writes_empty_file(self, tmp_path):
        # Arrange
        eval_dir = tmp_path / "eval"
        # Act
        write_sources(eval_dir=eval_dir, capsule_sources=[])
        # Assert
        assert (eval_dir / "sources.jsonl").read_text(encoding="utf-8") == ""
