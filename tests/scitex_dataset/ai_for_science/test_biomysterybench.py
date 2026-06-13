#!/usr/bin/env python3
"""Tests for ai_for_science.biomysterybench mask (no network)."""

import io
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import biomysterybench


def _build_csv_text(rows: list[dict]) -> str:
    """Render ``rows`` to CSV text without touching the filesystem.

    Used by the fixture so the fixture itself doesn't call ``open()`` —
    the audit's PA-307 STX-TQ005 rule flags fixtures that acquire an
    external resource via ``open()`` and return instead of yield. We
    keep CSV writing on an in-memory buffer to sidestep the pattern.
    """
    import csv as _csv

    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _sample_rows():
    return [
        {
            "id": "p1",
            "question": "Which organism is the model?",
            "allowed_domains": "biology",
            "human_solvable": "true",
            "answer_rubric": "The answer is Homo sapiens",
        },
        {
            "id": "p2",
            "question": "What gene is upregulated?",
            "allowed_domains": "biology",
            "human_solvable": "true",
            "answer_rubric": "The answer is BRCA1",
        },
    ]


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) leaves raw/."""
    base = tmp_path / "ai-for-science" / "biomysterybench" / "raw"
    base.mkdir(parents=True)
    (base / "problems.csv").write_text(_build_csv_text(_sample_rows()))
    return base


class TestMaskRow:
    def test_mask_row_nulls_answer_rubric(self):
        # Arrange
        row = {
            "id": "p1",
            "question": "Which gene?",
            "allowed_domains": "bio",
            "human_solvable": "true",
            "answer_rubric": "the answer is X",
        }
        # Act
        masked = biomysterybench.mask_row(row)
        # Assert
        assert masked["answer_rubric"] is None

    def test_mask_row_preserves_id(self):
        # Arrange
        row = {
            "id": "p1",
            "question": "Which gene?",
            "allowed_domains": "bio",
            "human_solvable": "true",
            "answer_rubric": "the answer is X",
        }
        # Act
        masked = biomysterybench.mask_row(row)
        # Assert
        assert masked["id"] == "p1"

    def test_mask_row_preserves_question(self):
        # Arrange
        row = {
            "id": "p1",
            "question": "Which gene?",
            "allowed_domains": "bio",
            "human_solvable": "true",
            "answer_rubric": "the answer is X",
        }
        # Act
        masked = biomysterybench.mask_row(row)
        # Assert
        assert masked["question"] == "Which gene?"

    def test_mask_row_drops_unknown_fields(self):
        # Arrange — upstream schema-drifted with an extra answer column
        row = {
            "id": "p1",
            "question": "Q",
            "allowed_domains": "bio",
            "human_solvable": "true",
            "answer_rubric": "the answer is X",
            "answer_hint": "starts with H",
        }
        # Act
        masked = biomysterybench.mask_row(row)
        # Assert
        assert "answer_hint" not in masked

    def test_mask_row_is_idempotent(self):
        # Arrange
        row = {
            "id": "p1",
            "question": "Q",
            "allowed_domains": "bio",
            "human_solvable": "true",
            "answer_rubric": "the answer is X",
        }
        once = biomysterybench.mask_row(row)
        # Act
        twice = biomysterybench.mask_row(once)
        # Assert
        assert once == twice


class TestMaskOnDisk:
    def test_mask_writes_questions_jsonl_in_masked_dir(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "questions.jsonl").is_file()

    def test_mask_record_count_matches_csv_row_count(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert result["n_records"] == 2

    def test_mask_creates_problems_masked_compat_symlink(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "problems_masked.jsonl").is_symlink()

    def test_mask_raises_when_problems_csv_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            biomysterybench.mask(raw_dir=bare, masked_dir=tmp_path / "out")


class TestMaskSymlinkView:
    def test_mask_symlinks_answer_free_data_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange — stage an answer-free problem environment alongside the CSV.
        (staged_raw_dir / "data").mkdir()
        (staged_raw_dir / "data" / "env.zip").write_text("zip-bytes")
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        link = masked_dir / "data"
        assert link.is_symlink() and link.resolve() == (staged_raw_dir / "data")

    def test_mask_does_not_symlink_oracle_csv_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert not (masked_dir / "problems.csv").is_symlink()

    def test_mask_result_symlinked_list_is_non_empty(self, tmp_path, staged_raw_dir):
        # Arrange — answer-free content present so a link is created.
        (staged_raw_dir / "data").mkdir()
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = biomysterybench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert len(result["symlinked"]) >= 1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
