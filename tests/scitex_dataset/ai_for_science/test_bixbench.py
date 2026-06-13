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
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) would leave raw/."""
    base = tmp_path / "ai-for-science" / "bixbench" / "raw"
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


class TestMaskOnDisk:
    def test_mask_writes_questions_jsonl_in_masked_dir(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "questions.jsonl").is_file()

    def test_mask_records_count_matches_raw_line_count(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert result["n_records"] == 1

    def test_mask_creates_backward_compat_symlink(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "BixBench_masked.jsonl").is_symlink()

    def test_mask_raises_when_raw_jsonl_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            bixbench.mask(raw_dir=bare, masked_dir=tmp_path / "out")


class TestMaskSymlinkView:
    def test_mask_symlinks_answer_free_capsule_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange — stage an answer-free capsule dir alongside the oracle.
        (staged_raw_dir / "CapsuleData-abc").mkdir()
        (staged_raw_dir / "CapsuleData-abc" / "data.csv").write_text("x,y\n1,2\n")
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        link = masked_dir / "CapsuleData-abc"
        assert link.is_symlink() and link.resolve() == (
            staged_raw_dir / "CapsuleData-abc"
        )

    def test_mask_does_not_symlink_oracle_manifest_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert not (masked_dir / "BixBench.jsonl").is_symlink()

    def test_mask_result_symlinked_list_is_non_empty(self, tmp_path, staged_raw_dir):
        # Arrange — answer-free content present so a link is created.
        (staged_raw_dir / "CapsuleData-abc").mkdir()
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert len(result["symlinked"]) >= 1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
