#!/usr/bin/env python3
"""Tests for ai_for_science.bixbench mask (no network)."""

import json
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import bixbench


def _sample_record():
    return {
        "id": "rec-1",
        "tag": "Q",
        "version": "v1",
        "question": "Which gene rises?",
        "hypothesis": "Truncating ASXL1 alters expression.",
        "capsule_uuid": "abc-123",
        "short_id": "sh1",
        "question_id": "qid1",
        "categories": ["bio"],
        "data_folder": "CapsuleData-abc",
        "eval_mode": "open",
        "canary": "CANARY-SENTINEL",
        # Oracle fields that must get nulled:
        "answer": "ASXL1",
        "ideal": "gene X",
        "result": "p<0.01",
        "distractors": ["gene Y", "gene Z"],
        "paper": "https://doi.org/10.1234/abc",
    }


@pytest.fixture
def oracle_record():
    return _sample_record()


@pytest.fixture
def staged_oracle_dir(tmp_path):
    base = tmp_path / "oracles" / "cohort_b_bixbench"
    base.mkdir(parents=True)
    (base / "BixBench.jsonl").write_text(json.dumps(_sample_record()) + "\n")
    return base


class TestMaskRecord:
    def test_mask_record_nulls_answer(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["answer"] is None

    def test_mask_record_nulls_ideal(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["ideal"] is None

    def test_mask_record_nulls_result(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["result"] is None

    def test_mask_record_nulls_distractors(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["distractors"] is None

    def test_mask_record_nulls_paper(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["paper"] is None

    def test_mask_record_preserves_hypothesis(self, oracle_record):
        # Arrange
        expected = oracle_record["hypothesis"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["hypothesis"] == expected

    def test_mask_record_preserves_canary(self, oracle_record):
        # Arrange
        expected = oracle_record["canary"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["canary"] == expected

    def test_mask_record_preserves_question(self, oracle_record):
        # Arrange
        expected = oracle_record["question"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["question"] == expected

    def test_mask_record_is_idempotent(self, oracle_record):
        # Arrange
        once = bixbench.mask_record(oracle_record)
        # Act
        twice = bixbench.mask_record(once)
        # Assert
        assert once == twice

    def test_mask_record_does_not_mutate_input(self, oracle_record):
        # Arrange
        snapshot = json.loads(json.dumps(oracle_record))
        # Act
        _ = bixbench.mask_record(oracle_record)
        # Assert
        assert oracle_record == snapshot


class TestRelocateOracleManifest:
    def test_relocate_moves_local_to_oracle_when_only_local_present(
        self, tmp_path
    ):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        local.parent.mkdir()
        local.write_text("payload\n")
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        # Act
        status = bixbench._relocate_oracle_manifest(local, oracle)
        # Assert
        assert status == "moved-local-to-oracle"

    def test_relocate_clears_local_after_moving_to_oracle(self, tmp_path):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        local.parent.mkdir()
        local.write_text("payload\n")
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        # Act
        bixbench._relocate_oracle_manifest(local, oracle)
        # Assert
        assert not local.exists()

    def test_relocate_returns_already_relocated_when_only_oracle_present(
        self, tmp_path
    ):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        oracle.parent.mkdir()
        oracle.write_text("payload\n")
        # Act
        status = bixbench._relocate_oracle_manifest(local, oracle)
        # Assert
        assert status == "already-relocated"

    def test_relocate_removes_local_duplicate_when_both_match(self, tmp_path):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        local.parent.mkdir()
        local.write_text("same-bytes\n")
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        oracle.parent.mkdir()
        oracle.write_text("same-bytes\n")
        # Act
        status = bixbench._relocate_oracle_manifest(local, oracle)
        # Assert
        assert status == "removed-duplicate-local-copy"

    def test_relocate_raises_runtimeerror_when_both_present_and_differ(
        self, tmp_path
    ):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        local.parent.mkdir()
        local.write_text("local-bytes\n")
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        oracle.parent.mkdir()
        oracle.write_text("oracle-bytes\n")
        # Act
        # Assert
        with pytest.raises(RuntimeError):
            bixbench._relocate_oracle_manifest(local, oracle)

    def test_relocate_raises_filenotfounderror_when_both_absent(self, tmp_path):
        # Arrange
        local = tmp_path / "capsule" / "BixBench.jsonl"
        oracle = tmp_path / "oracle" / "BixBench.jsonl"
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            bixbench._relocate_oracle_manifest(local, oracle)


class TestMaskOnDisk:
    def test_mask_writes_questions_jsonl_in_benchmark_dir(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        bixbench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert (benchmark_dir / "questions.jsonl").is_file()

    def test_mask_records_count_matches_oracle_line_count(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        result = bixbench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert result["n_records"] == 1

    def test_mask_creates_backward_compat_symlink(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        bixbench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert (benchmark_dir / "BixBench_masked.jsonl").is_symlink()

    def test_mask_raises_when_oracle_jsonl_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            bixbench.mask(oracle_dir=bare, benchmark_dir=tmp_path / "out")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
