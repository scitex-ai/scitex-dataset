#!/usr/bin/env python3
"""Smoke tests for ai_for_science._base path resolution."""

from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _base


class TestResolvePaths:
    def test_resolve_paths_returns_benchmark_paths_dataclass(self, tmp_path):
        # Arrange
        cohort_dir = "cohort_x_test"
        # Act
        paths = _base.resolve_paths(
            cohort_dir,
            oracle_root=tmp_path / "oracles",
            dataset_root=tmp_path / "dataset",
        )
        # Assert
        assert isinstance(paths, _base.BenchmarkPaths)

    def test_resolve_paths_oracle_dir_lives_under_oracle_root(self, tmp_path):
        # Arrange
        cohort_dir = "cohort_x_test"
        oracle_root = tmp_path / "oracles"
        # Act
        paths = _base.resolve_paths(
            cohort_dir,
            oracle_root=oracle_root,
            dataset_root=tmp_path / "dataset",
        )
        # Assert
        assert paths.oracle_dir == oracle_root / cohort_dir

    def test_resolve_paths_capsule_dir_under_dataset_root_src_capsules(
        self, tmp_path
    ):
        # Arrange
        cohort_dir = "cohort_x_test"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(
            cohort_dir, oracle_root=tmp_path, dataset_root=dataset_root
        )
        # Assert
        assert (
            paths.capsule_dir == dataset_root / cohort_dir / "src" / "capsules"
        )

    def test_resolve_paths_benchmark_dir_under_dataset_root_src_benchmark(
        self, tmp_path
    ):
        # Arrange
        cohort_dir = "cohort_x_test"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(
            cohort_dir, oracle_root=tmp_path, dataset_root=dataset_root
        )
        # Assert
        assert (
            paths.benchmark_dir
            == dataset_root / cohort_dir / "src" / "benchmark"
        )

    def test_resolve_paths_manifest_dir_under_cohort_scitex_dataset(
        self, tmp_path
    ):
        # Arrange
        cohort_dir = "cohort_x_test"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(
            cohort_dir, oracle_root=tmp_path, dataset_root=dataset_root
        )
        # Assert
        assert (
            paths.manifest_dir
            == dataset_root / cohort_dir / ".scitex" / "dataset"
        )

    def test_benchmark_paths_as_dict_returns_string_values(self, tmp_path):
        # Arrange
        cohort_dir = "cohort_x_test"
        paths = _base.resolve_paths(
            cohort_dir,
            oracle_root=tmp_path / "oracles",
            dataset_root=tmp_path / "dataset",
        )
        # Act
        d = paths.as_dict()
        # Assert
        assert all(isinstance(v, str) for v in d.values())

    def test_resolve_paths_honours_oracles_root_env_var(self, tmp_path):
        # Arrange — manage env via os.environ with try/finally so the
        # test stays mock-free (PA-306 forbids the monkeypatch fixture).
        import os

        env_root = tmp_path / "env-oracles"
        prior = os.environ.get("SCITEX_ORACLES_ROOT")
        os.environ["SCITEX_ORACLES_ROOT"] = str(env_root)
        try:
            # Act
            paths = _base.resolve_paths("cohort_y_test")
        finally:
            if prior is None:
                os.environ.pop("SCITEX_ORACLES_ROOT", None)
            else:
                os.environ["SCITEX_ORACLES_ROOT"] = prior
        # Assert
        assert paths.oracle_dir == env_root / "cohort_y_test"

    def test_resolve_paths_honours_dataset_root_env_var(self, tmp_path):
        # Arrange — manage env via os.environ with try/finally so the
        # test stays mock-free (PA-306 forbids the monkeypatch fixture).
        import os

        env_root = tmp_path / "env-dataset"
        prior = os.environ.get("SCITEX_DATASET_ROOT")
        os.environ["SCITEX_DATASET_ROOT"] = str(env_root)
        try:
            # Act
            paths = _base.resolve_paths("cohort_z_test")
        finally:
            if prior is None:
                os.environ.pop("SCITEX_DATASET_ROOT", None)
            else:
                os.environ["SCITEX_DATASET_ROOT"] = prior
        # Assert
        assert (
            paths.capsule_dir
            == env_root / "cohort_z_test" / "src" / "capsules"
        )


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
