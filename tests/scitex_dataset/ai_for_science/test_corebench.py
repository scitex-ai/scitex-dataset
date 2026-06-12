#!/usr/bin/env python3
"""Tests for ai_for_science.corebench mask + inventory (no network)."""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import corebench


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_train_record():
    return {
        "capsule_id": "capsule-1111111",
        "language": "Python",
        "field": "biology",
        "task_prompt": "Read the README.",
        "capsule_title": "A Paper About Stuff",
        "capsule_doi": "10.1234/foo",
        "results": [
            {"What is the AUC?": 0.81},
            {"How many trials?": 12},
            {"What pH?": 7.4},
        ],
    }


def _make_test_record():
    return {
        "capsule_id": "capsule-2222222",
        "language": "R",
        "field": "ecology",
        "task_prompt": "Run the R script.",
        "capsule_title": "Another Paper",
        "capsule_doi": "10.5678/bar",
        "results": [
            {"What is the mean?": 4.2},
            {"What is the variance?": 0.5},
        ],
    }


@pytest.fixture
def oracle_train_record():
    return _make_train_record()


@pytest.fixture
def oracle_test_record():
    return _make_test_record()


@pytest.fixture
def staged_oracle_dir(tmp_path):
    """Lay out the oracle directory the same way download.sh would."""
    base = tmp_path / "oracles" / "cohort_a_corebench"
    (base / "dataset").mkdir(parents=True)
    (base / "dataset" / "core_train.json").write_text(
        json.dumps([_make_train_record()])
    )
    (base / "core_test.json").write_text(json.dumps([_make_test_record()]))
    return base


# ---------------------------------------------------------------------------
# mask_record (unit) — one assertion per test
# ---------------------------------------------------------------------------


class TestMaskRecord:
    def test_mask_record_nulls_every_answer_value(self, oracle_train_record):
        # Arrange
        rec = oracle_train_record
        # Act
        masked = corebench.mask_record(rec)
        # Assert
        assert all(
            v is None for entry in masked["results"] for v in entry.values()
        )

    def test_mask_record_preserves_question_keys(self, oracle_train_record):
        # Arrange
        original_keys = [list(e.keys())[0] for e in oracle_train_record["results"]]
        # Act
        masked = corebench.mask_record(oracle_train_record)
        # Assert
        assert [list(e.keys())[0] for e in masked["results"]] == original_keys

    def test_mask_record_nulls_capsule_title(self, oracle_train_record):
        # Arrange
        rec = oracle_train_record
        # Act
        masked = corebench.mask_record(rec)
        # Assert
        assert masked["capsule_title"] is None

    def test_mask_record_nulls_capsule_doi(self, oracle_train_record):
        # Arrange
        rec = oracle_train_record
        # Act
        masked = corebench.mask_record(rec)
        # Assert
        assert masked["capsule_doi"] is None

    def test_mask_record_preserves_capsule_id(self, oracle_train_record):
        # Arrange
        expected = oracle_train_record["capsule_id"]
        # Act
        masked = corebench.mask_record(oracle_train_record)
        # Assert
        assert masked["capsule_id"] == expected

    def test_mask_record_preserves_task_prompt(self, oracle_train_record):
        # Arrange
        expected = oracle_train_record["task_prompt"]
        # Act
        masked = corebench.mask_record(oracle_train_record)
        # Assert
        assert masked["task_prompt"] == expected

    def test_mask_record_does_not_mutate_input(self, oracle_train_record):
        # Arrange
        snapshot = json.loads(json.dumps(oracle_train_record))
        # Act
        _ = corebench.mask_record(oracle_train_record)
        # Assert
        assert oracle_train_record == snapshot

    def test_mask_record_is_idempotent_after_double_apply(
        self, oracle_train_record
    ):
        # Arrange
        once = corebench.mask_record(oracle_train_record)
        # Act
        twice = corebench.mask_record(once)
        # Assert
        assert once == twice


# ---------------------------------------------------------------------------
# mask (orchestration on disk)
# ---------------------------------------------------------------------------


class TestMaskOnDisk:
    def test_mask_emits_questions_json_under_benchmark_dir(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        corebench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert (benchmark_dir / "questions.json").is_file()

    def test_mask_returns_n_records_two_for_train_plus_test_singletons(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        result = corebench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert result["n_records"] == 2

    def test_mask_records_n_train_count_separately(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        result = corebench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert result["n_train"] == 1

    def test_mask_records_n_test_count_separately(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        result = corebench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert result["n_test"] == 1

    def test_mask_raises_when_oracle_train_json_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty-oracle"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            corebench.mask(oracle_dir=bare, benchmark_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# build_inventory
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def test_build_inventory_writes_inventory_json_next_to_capsules(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        capsule_dir = tmp_path / "data" / "src" / "capsules"
        # Act
        corebench.build_inventory(
            oracle_dir=staged_oracle_dir, capsule_dir=capsule_dir
        )
        # Assert
        assert (capsule_dir.parent / "inventory.json").is_file()

    def test_build_inventory_summary_counts_one_train_capsule(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        capsule_dir = tmp_path / "data" / "src" / "capsules"
        # Act
        result = corebench.build_inventory(
            oracle_dir=staged_oracle_dir, capsule_dir=capsule_dir
        )
        # Assert
        assert result["summary"]["n_capsules_train"] == 1

    def test_build_inventory_summary_counts_one_test_capsule(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        capsule_dir = tmp_path / "data" / "src" / "capsules"
        # Act
        result = corebench.build_inventory(
            oracle_dir=staged_oracle_dir, capsule_dir=capsule_dir
        )
        # Assert
        assert result["summary"]["n_capsules_test"] == 1


# ---------------------------------------------------------------------------
# prepare orchestrator (skip_download path — no network)
# ---------------------------------------------------------------------------


class TestClassifyCapsuleFiles:
    def test_classify_returns_none_when_capsule_dir_absent(self, tmp_path):
        # Arrange
        missing = tmp_path / "nope"
        # Act
        result = corebench._classify_capsule_files(missing)
        # Assert
        assert result is None

    def test_classify_counts_python_files(self, tmp_path):
        # Arrange
        capsule = tmp_path / "capsule-x"
        capsule.mkdir()
        (capsule / "a.py").write_text("# a")
        (capsule / "b.py").write_text("# b")
        # Act
        stats = corebench._classify_capsule_files(capsule)
        # Assert
        assert stats["n_python_files"] == 2

    def test_classify_counts_r_files_case_insensitive(self, tmp_path):
        # Arrange
        capsule = tmp_path / "capsule-x"
        capsule.mkdir()
        (capsule / "analysis.R").write_text("# r")
        (capsule / "report.rmd").write_text("# rmd")
        # Act
        stats = corebench._classify_capsule_files(capsule)
        # Assert
        assert stats["n_r_files"] == 2

    def test_classify_counts_notebooks(self, tmp_path):
        # Arrange
        capsule = tmp_path / "capsule-x"
        capsule.mkdir()
        (capsule / "nb.ipynb").write_text("{}")
        # Act
        stats = corebench._classify_capsule_files(capsule)
        # Assert
        assert stats["n_notebooks"] == 1

    def test_classify_detects_dockerfile(self, tmp_path):
        # Arrange
        capsule = tmp_path / "capsule-x"
        capsule.mkdir()
        (capsule / "Dockerfile").write_text("FROM python")
        # Act
        stats = corebench._classify_capsule_files(capsule)
        # Assert
        assert stats["has_dockerfile"] is True

    def test_classify_counts_data_files_by_extension(self, tmp_path):
        # Arrange
        capsule = tmp_path / "capsule-x"
        capsule.mkdir()
        for name in ("a.csv", "b.parquet", "c.npy"):
            (capsule / name).write_text("x")
        # Act
        stats = corebench._classify_capsule_files(capsule)
        # Assert
        assert stats["n_data_files"] == 3


class TestCapsuleUrl:
    def test_capsule_url_combines_base_and_capsule_id(self):
        # Arrange
        base = "https://corebench.cs.princeton.edu"
        cid = "capsule-7038571"
        # Act
        url = corebench._capsule_url(base, cid)
        # Assert
        assert url == f"{base}/capsules/{cid}.tar.gz"


class TestDownloadSkipsExistingCapsules:
    def test_download_marks_existing_file_as_have(self, tmp_path):
        # Arrange — pre-stage a capsule tarball so download(...) sees
        # it as already present (no network call attempted).
        capsule_dir = tmp_path / "cap"
        capsule_dir.mkdir()
        (capsule_dir / "capsule-abc.tar.gz").write_text("not-really-a-tarball")
        # Act
        result = corebench.download(
            oracle_dir=tmp_path / "oracle",
            capsule_dir=capsule_dir,
            capsule_ids=["capsule-abc"],
        )
        # Assert
        assert result["n_have"] == 1

    def test_download_records_zero_fetched_when_only_existing_present(
        self, tmp_path
    ):
        # Arrange
        capsule_dir = tmp_path / "cap"
        capsule_dir.mkdir()
        (capsule_dir / "capsule-abc.tar.gz").write_text("x")
        # Act
        result = corebench.download(
            oracle_dir=tmp_path / "oracle",
            capsule_dir=capsule_dir,
            capsule_ids=["capsule-abc"],
        )
        # Assert
        assert result["n_fetched"] == 0

    def test_download_raises_when_capsule_ids_none_and_oracle_missing(
        self, tmp_path
    ):
        # Arrange
        capsule_dir = tmp_path / "cap"
        bare_oracle = tmp_path / "empty-oracle"
        bare_oracle.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            corebench.download(
                oracle_dir=bare_oracle, capsule_dir=capsule_dir
            )


class TestPrepareSkipDownload:
    def test_prepare_skip_download_emits_manifest_yaml(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        from scitex_dataset.ai_for_science import _base

        paths = _base.BenchmarkPaths(
            cohort_dir=corebench.COHORT_DIR,
            oracle_dir=staged_oracle_dir,
            capsule_dir=tmp_path / "data" / "src" / "capsules",
            benchmark_dir=tmp_path / "data" / "src" / "benchmark",
            manifest_dir=tmp_path / "data" / ".scitex" / "dataset",
        )
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert Path(result["manifest"]).is_file()

    def test_prepare_skip_download_manifest_starts_with_corebench_id(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        from scitex_dataset.ai_for_science import _base

        paths = _base.BenchmarkPaths(
            cohort_dir=corebench.COHORT_DIR,
            oracle_dir=staged_oracle_dir,
            capsule_dir=tmp_path / "data" / "src" / "capsules",
            benchmark_dir=tmp_path / "data" / "src" / "benchmark",
            manifest_dir=tmp_path / "data" / ".scitex" / "dataset",
        )
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert Path(result["manifest"]).read_text().startswith("id: corebench")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
