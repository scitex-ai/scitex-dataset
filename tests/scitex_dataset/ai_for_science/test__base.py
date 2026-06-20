#!/usr/bin/env python3
"""Smoke tests for ai_for_science._base path resolution.

Contract: raw -> {for_solver, eval} (no masked_dir).
"""

from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _base


class TestResolvePaths:
    def test_resolve_paths_returns_benchmark_paths_dataclass(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=tmp_path / "dataset")
        # Assert
        assert isinstance(paths, _base.BenchmarkPaths)

    def test_resolve_paths_root_lives_under_ai_for_science_benchmark(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=dataset_root)
        # Assert
        assert paths.root == dataset_root / "ai-for-science" / benchmark

    def test_resolve_paths_raw_dir_under_benchmark_root(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=dataset_root)
        # Assert
        assert paths.raw_dir == dataset_root / "ai-for-science" / benchmark / "raw"

    def test_resolve_paths_for_solver_dir_under_benchmark_root(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=dataset_root)
        # Assert
        assert (
            paths.for_solver_dir
            == dataset_root / "ai-for-science" / benchmark / "for_solver"
        )

    def test_resolve_paths_eval_dir_under_benchmark_root(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=dataset_root)
        # Assert
        assert paths.eval_dir == dataset_root / "ai-for-science" / benchmark / "eval"

    def test_resolve_paths_has_no_masked_dir_field(self, tmp_path):
        # Arrange
        paths = _base.resolve_paths("corebench", dataset_root=tmp_path / "dataset")
        # Act
        keys = set(paths.as_dict())
        # Assert
        assert "masked_dir" not in keys

    def test_resolve_paths_manifest_dir_under_benchmark_scitex_dataset(self, tmp_path):
        # Arrange
        benchmark = "corebench"
        dataset_root = tmp_path / "dataset"
        # Act
        paths = _base.resolve_paths(benchmark, dataset_root=dataset_root)
        # Assert
        assert (
            paths.manifest_dir
            == dataset_root / "ai-for-science" / benchmark / ".scitex" / "dataset"
        )

    def test_benchmark_paths_as_dict_returns_string_values(self, tmp_path):
        # Arrange
        paths = _base.resolve_paths("corebench", dataset_root=tmp_path / "dataset")
        # Act
        d = paths.as_dict()
        # Assert
        assert all(isinstance(v, str) for v in d.values())

    def test_domain_constant_is_ai_for_science(self):
        # Arrange
        module = _base
        # Act
        domain = module.DOMAIN
        # Assert
        assert domain == "ai-for-science"

    def test_resolve_paths_honours_dataset_root_env_var(self, tmp_path):
        # Arrange — manage env via os.environ with try/finally so the
        # test stays mock-free (PA-306 forbids the monkeypatch fixture).
        import os

        env_root = tmp_path / "env-dataset"
        prior = os.environ.get("SCITEX_DATASET_ROOT")
        os.environ["SCITEX_DATASET_ROOT"] = str(env_root)
        try:
            # Act
            paths = _base.resolve_paths("bixbench")
        finally:
            if prior is None:
                os.environ.pop("SCITEX_DATASET_ROOT", None)
            else:
                os.environ["SCITEX_DATASET_ROOT"] = prior
        # Assert
        assert paths.raw_dir == env_root / "ai-for-science" / "bixbench" / "raw"

    def test_resolve_paths_env_var_lands_under_ai_for_science_benchmark(self, tmp_path):
        # Arrange
        import os

        env_root = tmp_path / "env-dataset2"
        prior = os.environ.get("SCITEX_DATASET_ROOT")
        os.environ["SCITEX_DATASET_ROOT"] = str(env_root)
        try:
            # Act
            paths = _base.resolve_paths("biomysterybench")
        finally:
            if prior is None:
                os.environ.pop("SCITEX_DATASET_ROOT", None)
            else:
                os.environ["SCITEX_DATASET_ROOT"] = prior
        # Assert
        assert paths.root == env_root / "ai-for-science" / "biomysterybench"

    def test_resolve_paths_falls_back_to_config_chain_when_no_override_or_env(self):
        # Arrange — with neither an explicit override NOR the env var, the
        # resolver falls through to the _config project→user chain
        # (``project_root() or user_root()``). We don't assert the exact
        # path (it depends on where pytest runs) — only that resolution
        # succeeds and lands under the ai-for-science category dir. This
        # exercises the final fallback branch (_base.py:71) that the
        # override/env tests skip. Env managed via os.environ save/restore
        # so the test stays mock-free (PA-306 forbids the monkeypatch fixture).
        import os

        prior = os.environ.get("SCITEX_DATASET_ROOT")
        os.environ.pop("SCITEX_DATASET_ROOT", None)
        try:
            # Act
            paths = _base.resolve_paths("corebench")
        finally:
            if prior is not None:
                os.environ["SCITEX_DATASET_ROOT"] = prior
        # Assert
        assert paths.root.parent.name == "ai-for-science"

    def test_resolve_paths_fallback_root_ends_with_benchmark_name(self):
        # Arrange
        import os

        prior = os.environ.get("SCITEX_DATASET_ROOT")
        os.environ.pop("SCITEX_DATASET_ROOT", None)
        try:
            # Act
            paths = _base.resolve_paths("corebench")
        finally:
            if prior is not None:
                os.environ["SCITEX_DATASET_ROOT"] = prior
        # Assert
        assert paths.root.name == "corebench"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
