#!/usr/bin/env python3
"""Tests for ai_for_science._sources (per-capsule source registration).

No mocks / monkeypatch: every check uses real relpaths and real tmp
trees on disk. Covers the doc denylist, the source/doc split, the
filesystem snapshot, and the ``sources.jsonl`` writer (incl. idempotence).
"""

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science._sources import (
    _denest_single_top,
    classify_capsule_sources,
    is_doc_source,
    list_archive_members,
    register_capsule_sources,
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


# ---------------------------------------------------------------------------
# Archive helpers (real .tar.gz / .zip on disk — no mocks)
# ---------------------------------------------------------------------------


def _write_targz(archive_path: Path, files: dict) -> None:
    with tarfile.open(archive_path, "w:gz") as tf:
        for arcname, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _write_zip(archive_path: Path, files: dict) -> None:
    with zipfile.ZipFile(archive_path, "w") as zf:
        for arcname, content in files.items():
            zf.writestr(arcname, content)


_MEMBER_FILES = {
    "data/raw.csv": "x",
    "code/main.py": "y",
    "results/out.csv": "z",
    "README.md": "doc",
}


# ---------------------------------------------------------------------------
# list_archive_members — member listing without extraction
# ---------------------------------------------------------------------------


class TestListArchiveMembers:
    @pytest.mark.parametrize(
        "suffix, writer",
        [(".tar.gz", _write_targz), (".zip", _write_zip)],
    )
    def test_lists_sorted_file_members_without_dir_entries(
        self, tmp_path, suffix, writer
    ):
        # Arrange
        archive = tmp_path / f"cap{suffix}"
        writer(archive, _MEMBER_FILES)
        # Act
        members = list_archive_members(archive)
        # Assert
        assert members == [
            "README.md",
            "code/main.py",
            "data/raw.csv",
            "results/out.csv",
        ]

    def test_missing_archive_raises_file_not_found(self, tmp_path):
        # Arrange
        archive = tmp_path / "absent.tar.gz"
        # Act
        call = lambda: list_archive_members(archive)  # noqa: E731
        # Assert
        with pytest.raises(FileNotFoundError):
            call()

    def test_unrecognised_suffix_raises_value_error(self, tmp_path):
        # Arrange
        archive = tmp_path / "cap.rar"
        archive.write_text("not-an-archive", encoding="utf-8")
        # Act
        call = lambda: list_archive_members(archive)  # noqa: E731
        # Assert
        with pytest.raises(ValueError):
            call()


# ---------------------------------------------------------------------------
# _denest_single_top — member-list de-nesting
# ---------------------------------------------------------------------------


class TestDenestSingleTop:
    def test_single_wrapping_dir_is_denested(self):
        # Arrange
        names = ["cap-1/data/x.csv", "cap-1/README.md"]
        # Act
        out = _denest_single_top(names)
        # Assert
        assert out == ["README.md", "data/x.csv"]

    def test_multiple_top_segments_are_unchanged(self):
        # Arrange
        names = ["data/x", "code/y"]
        # Act
        out = _denest_single_top(names)
        # Assert
        assert out == ["code/y", "data/x"]

    def test_top_level_file_blocks_denest(self):
        # Arrange — one top FILE + one top dir: not a single wrapping dir.
        names = ["foo.txt", "bar/baz"]
        # Act
        out = _denest_single_top(names)
        # Assert
        assert out == ["bar/baz", "foo.txt"]

    def test_empty_names_returns_empty(self):
        # Arrange
        names = []
        # Act
        out = _denest_single_top(names)
        # Assert
        assert out == []


# ---------------------------------------------------------------------------
# register_capsule_sources — raw-archive member listing in the callers
# ---------------------------------------------------------------------------


def _register_fixture(tmp_path, *, with_missing=False):
    """Build a raw_dir with a real wrapped archive + tasks; return context."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "capsules").mkdir(parents=True)
    _write_targz(
        raw_dir / "capsules" / "cap-1.tar.gz",
        {
            "cap-1/data/raw.csv": "x",
            "cap-1/code/main.py": "y",
            "cap-1/results/out.csv": "z",
            "cap-1/README.md": "doc",
            "cap-1/REPRODUCING.md": "doc",
        },
    )
    tasks = [
        {
            "task_id": "corebench/cap-1__hard__q0",
            "benchmark": "corebench",
            "prompt": "p",
            "data": "./capsules/cap-1.tar.gz",
        }
    ]
    if with_missing:
        tasks.append(
            {
                "task_id": "corebench/cap-2__hard__q0",
                "benchmark": "corebench",
                "prompt": "p",
                "data": "./capsules/cap-2.tar.gz",  # archive not written
            }
        )
    eval_dir = tmp_path / "eval"
    return raw_dir, tasks, eval_dir


def _written_rows(eval_dir):
    lines = (eval_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class TestRegisterCapsuleSources:
    def test_register_writes_one_parseable_row_per_present_capsule(self, tmp_path):
        # Arrange
        raw_dir, tasks, eval_dir = _register_fixture(tmp_path)
        # Act
        register_capsule_sources(tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir)
        # Assert
        assert len(_written_rows(eval_dir)) == 1

    def test_register_excludes_readme_and_reproducing_docs(self, tmp_path):
        # Arrange
        raw_dir, tasks, eval_dir = _register_fixture(tmp_path)
        # Act
        register_capsule_sources(tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir)
        # Assert
        assert _written_rows(eval_dir)[0]["excluded_docs"] == [
            "README.md",
            "REPRODUCING.md",
        ]

    def test_register_keeps_data_code_and_results_as_sources(self, tmp_path):
        # Arrange
        raw_dir, tasks, eval_dir = _register_fixture(tmp_path)
        # Act
        register_capsule_sources(tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir)
        # Assert
        assert _written_rows(eval_dir)[0]["sources"] == [
            "code/main.py",
            "data/raw.csv",
            "results/out.csv",
        ]

    def test_register_reports_missing_archive_native_id(self, tmp_path):
        # Arrange
        raw_dir, tasks, eval_dir = _register_fixture(tmp_path, with_missing=True)
        # Act
        ret = register_capsule_sources(
            tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir
        )
        # Assert
        assert ret["missing"] == ["cap-2"]

    def test_register_omits_missing_capsule_from_sources_jsonl(self, tmp_path):
        # Arrange
        raw_dir, tasks, eval_dir = _register_fixture(tmp_path, with_missing=True)
        # Act
        register_capsule_sources(tasks=tasks, raw_dir=raw_dir, eval_dir=eval_dir)
        # Assert
        assert [r["native_id"] for r in _written_rows(eval_dir)] == ["cap-1"]
