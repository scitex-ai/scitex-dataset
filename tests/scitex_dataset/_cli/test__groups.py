#!/usr/bin/env python3
"""CLI smoke tests for the ai-for-science domain (no network)."""

import json
import zipfile

import pytest
from click.testing import CliRunner

from scitex_dataset._cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestDomainGroupWiring:
    def test_top_level_help_lists_ai_for_science_domain(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["-h"])
        # Assert
        assert "ai-for-science" in result.output

    def test_ai_for_science_help_lists_corebench(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", "-h"])
        # Assert
        assert "corebench" in result.output

    def test_ai_for_science_help_lists_bixbench(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", "-h"])
        # Assert
        assert "bixbench" in result.output

    def test_ai_for_science_help_lists_biomysterybench(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", "-h"])
        # Assert
        assert "biomysterybench" in result.output

    @pytest.mark.parametrize("bench", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_help_lists_download_verb(self, runner, bench):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", bench, "-h"])
        # Assert
        assert "download" in result.output

    @pytest.mark.parametrize("bench", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_help_lists_prepare_verb(self, runner, bench):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", bench, "-h"])
        # Assert
        assert "prepare" in result.output

    @pytest.mark.parametrize("bench", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_help_lists_standardize_verb(self, runner, bench):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", bench, "-h"])
        # Assert
        assert "standardize" in result.output


def _stage_bixbench_raw(dataset_root):
    """Stage a one-record BixBench raw/ snapshot under the dataset root.

    ``standardize`` writes the per-capsule layout: the record's
    ``data_folder`` archive is EXTRACTED into ``capsule-NNN/input/``, so a
    REAL ``.zip`` must sit at the path ``data`` points at.
    """
    raw_dir = dataset_root / "ai-for-science" / "bixbench" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "BixBench.jsonl").write_text(
        json.dumps(
            {
                "short_id": "x",
                "question_id": "x-q1",
                "question": "q?",
                "data_folder": "CapsuleFolder-x.zip",
                "answer": "secret",
                "ideal": "i",
                "canary": "c",
                "hypothesis": "h",
            }
        )
        + "\n"
    )
    with zipfile.ZipFile(raw_dir / "CapsuleFolder-x.zip", "w") as zf:
        zf.writestr("notebook/analysis.py", "# scaffold\n")
    return raw_dir


class TestStandardizeCli:
    def test_corebench_standardize_missing_raw_exits_nonzero(self, runner, tmp_path):
        # Arrange — point the dataset root at an empty tmp dir
        args = [
            "ai-for-science",
            "corebench",
            "standardize",
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code != 0

    def test_bixbench_standardize_with_staged_raw_exits_zero(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_bixbench_raw(dataset_root)
        args = [
            "ai-for-science",
            "bixbench",
            "standardize",
            "--dataset-root",
            str(dataset_root),
            "--json",
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code == 0

    def test_bixbench_standardize_emits_one_task_in_json_output(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_bixbench_raw(dataset_root)
        args = [
            "ai-for-science",
            "bixbench",
            "standardize",
            "--dataset-root",
            str(dataset_root),
            "--json",
        ]
        result = runner.invoke(main, args)
        # Act
        payload = json.loads(result.output)
        # Assert
        assert payload["n_tasks"] == 1


def _stage_corebench_eval(dataset_root):
    """Stage a one-answer corebench eval/answers.jsonl under the dataset root."""
    eval_dir = dataset_root / "ai-for-science" / "corebench" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "answers.jsonl").write_text(
        json.dumps(
            {
                "task_id": "corebench/capsule-1__hard__q0",
                "answer": {"value": 0.9996},
            }
        )
        + "\n"
    )
    return eval_dir


class TestValidateCli:
    def test_help_lists_validate_verb(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", "corebench", "-h"])
        # Assert
        assert "validate" in result.output

    def test_valid_submission_exits_zero(self, runner, tmp_path):
        # Arrange
        sub = tmp_path / "sub.json"
        sub.write_text(
            json.dumps(
                [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.9996}]
            )
        )
        args = [
            "ai-for-science",
            "corebench",
            "validate",
            "--submission",
            str(sub),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code == 0

    def test_invalid_submission_exits_nonzero(self, runner, tmp_path):
        # Arrange — a bad task_id makes it structurally invalid.
        sub = tmp_path / "sub.json"
        sub.write_text(json.dumps([{"task_id": "bixbench/x", "answer": 1}]))
        args = [
            "ai-for-science",
            "corebench",
            "validate",
            "--submission",
            str(sub),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code != 0


class TestScoreCli:
    def test_help_lists_score_verb(self, runner):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", "corebench", "-h"])
        # Assert
        assert "score" in result.output

    def test_score_emits_verdict_in_json(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_corebench_eval(dataset_root)
        sub = tmp_path / "sub.json"
        sub.write_text(
            json.dumps(
                [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
            )
        )
        args = [
            "ai-for-science",
            "corebench",
            "score",
            "--submission",
            str(sub),
            "--dataset-root",
            str(dataset_root),
            "--json",
        ]
        result = runner.invoke(main, args)
        # Act
        payload = json.loads(result.output)
        # Assert
        assert payload[0]["verdict"] == "correct"

    def test_score_exits_zero(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_corebench_eval(dataset_root)
        sub = tmp_path / "sub.json"
        sub.write_text(
            json.dumps(
                [{"task_id": "corebench/capsule-1__hard__q0", "answer": 0.99956}]
            )
        )
        args = [
            "ai-for-science",
            "corebench",
            "score",
            "--submission",
            str(sub),
            "--dataset-root",
            str(dataset_root),
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code == 0


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
