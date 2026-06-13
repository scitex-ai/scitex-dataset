#!/usr/bin/env python3
"""CLI smoke tests for the ai-for-science domain (no network)."""

import json

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
    def test_benchmark_help_lists_mask_verb(self, runner, bench):
        # Arrange
        cli = main
        # Act
        result = runner.invoke(cli, ["ai-for-science", bench, "-h"])
        # Assert
        assert "mask" in result.output


def _stage_bixbench_raw(dataset_root):
    """Stage a one-record BixBench raw/ snapshot under the dataset root."""
    raw_dir = dataset_root / "ai-for-science" / "bixbench" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "BixBench.jsonl").write_text(
        json.dumps(
            {
                "id": "x",
                "question": "q?",
                "answer": "secret",
                "ideal": "i",
                "result": "r",
                "distractors": [],
                "paper": "p",
                "canary": "c",
                "hypothesis": "h",
            }
        )
        + "\n"
    )
    return raw_dir


class TestMaskCli:
    def test_corebench_mask_missing_raw_exits_nonzero(self, runner, tmp_path):
        # Arrange — point the dataset root at an empty tmp dir
        args = [
            "ai-for-science",
            "corebench",
            "mask",
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code != 0

    def test_bixbench_mask_with_staged_raw_exits_zero(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_bixbench_raw(dataset_root)
        args = [
            "ai-for-science",
            "bixbench",
            "mask",
            "--dataset-root",
            str(dataset_root),
            "--json",
        ]
        # Act
        result = runner.invoke(main, args)
        # Assert
        assert result.exit_code == 0

    def test_bixbench_mask_emits_one_record_in_json_output(self, runner, tmp_path):
        # Arrange
        dataset_root = tmp_path / "dataset"
        _stage_bixbench_raw(dataset_root)
        args = [
            "ai-for-science",
            "bixbench",
            "mask",
            "--dataset-root",
            str(dataset_root),
            "--json",
        ]
        result = runner.invoke(main, args)
        # Act
        payload = json.loads(result.output)
        # Assert
        assert payload["n_records"] == 1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
