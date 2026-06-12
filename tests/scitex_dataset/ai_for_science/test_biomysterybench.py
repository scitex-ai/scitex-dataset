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
def staged_oracle_dir(tmp_path):
    base = tmp_path / "oracles" / "cohort_c_biomysterybench"
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
    def test_mask_writes_questions_jsonl_in_benchmark_dir(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        biomysterybench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert (benchmark_dir / "questions.jsonl").is_file()

    def test_mask_record_count_matches_csv_row_count(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        result = biomysterybench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert result["n_records"] == 2

    def test_mask_creates_problems_masked_compat_symlink(
        self, tmp_path, staged_oracle_dir
    ):
        # Arrange
        benchmark_dir = tmp_path / "bench"
        # Act
        biomysterybench.mask(
            oracle_dir=staged_oracle_dir, benchmark_dir=benchmark_dir
        )
        # Assert
        assert (benchmark_dir / "problems_masked.jsonl").is_symlink()

    def test_mask_raises_when_problems_csv_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            biomysterybench.mask(
                oracle_dir=bare, benchmark_dir=tmp_path / "out"
            )


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
