#!/usr/bin/env python3
"""Smoke tests for the ai_for_science package surface (no network)."""

import pytest

from scitex_dataset import ai_for_science


class TestPackageSurface:
    @pytest.mark.parametrize("name", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_submodule_is_importable(self, name):
        # Arrange
        module = ai_for_science
        # Act
        attr = getattr(module, name, None)
        # Assert
        assert attr is not None

    @pytest.mark.parametrize("name", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_submodule_exposes_standardize_callable(self, name):
        # Arrange
        module = getattr(ai_for_science, name)
        # Act
        standardize = getattr(module, "standardize", None)
        # Assert
        assert callable(standardize)

    @pytest.mark.parametrize("name", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_submodule_exposes_download_callable(self, name):
        # Arrange
        module = getattr(ai_for_science, name)
        # Act
        download = getattr(module, "download", None)
        # Assert
        assert callable(download)

    @pytest.mark.parametrize("name", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_submodule_exposes_prepare_callable(self, name):
        # Arrange
        module = getattr(ai_for_science, name)
        # Act
        prepare = getattr(module, "prepare", None)
        # Assert
        assert callable(prepare)

    @pytest.mark.parametrize("name", ["corebench", "bixbench", "biomysterybench"])
    def test_benchmark_submodule_declares_benchmark_identity(self, name):
        # Arrange
        module = getattr(ai_for_science, name)
        # Act
        benchmark = getattr(module, "BENCHMARK", "")
        # Assert
        assert benchmark == name

    def test_resolve_paths_is_reexported_from_package(self):
        # Arrange
        module = ai_for_science
        # Act
        attr = getattr(module, "resolve_paths", None)
        # Assert
        assert callable(attr)

    def test_write_manifest_is_reexported_from_package(self):
        # Arrange
        module = ai_for_science
        # Act
        attr = getattr(module, "write_manifest", None)
        # Assert
        assert callable(attr)

    def test_domain_is_reexported_from_package(self):
        # Arrange
        module = ai_for_science
        # Act
        domain = getattr(module, "DOMAIN", None)
        # Assert
        assert domain == "ai-for-science"

    @pytest.mark.parametrize(
        "name",
        ["DEFAULT_BENCHMARK_DIR", "DEFAULT_CAPSULE_DIR", "DEFAULT_ORACLE_DIR"],
    )
    def test_retired_default_dir_exports_are_gone(self, name):
        # Arrange
        module = ai_for_science
        # Act
        attr = getattr(module, name, None)
        # Assert
        assert attr is None


class TestPackageRootStandardizeAliases:
    @pytest.mark.parametrize(
        "name",
        [
            "corebench_standardize",
            "bixbench_standardize",
            "biomysterybench_standardize",
        ],
    )
    def test_standardize_alias_appears_in_package_root_all(self, name):
        # Arrange
        import scitex_dataset

        # Act
        listed = name in scitex_dataset.__all__
        # Assert
        assert listed

    @pytest.mark.parametrize(
        "name",
        [
            "corebench_standardize",
            "bixbench_standardize",
            "biomysterybench_standardize",
        ],
    )
    def test_standardize_alias_is_callable_on_package_root(self, name):
        # Arrange
        import scitex_dataset

        # Act
        attr = getattr(scitex_dataset, name, None)
        # Assert
        assert callable(attr)


class TestSourceRegistryEntries:
    @pytest.mark.parametrize("src", ["corebench", "bixbench", "biomysterybench"])
    def test_source_appears_in_agentic_bench_sources_list(self, src):
        # Arrange
        from scitex_dataset._sources import AGENTIC_BENCH_SOURCES

        # Act
        present = src in AGENTIC_BENCH_SOURCES
        # Assert
        assert present

    @pytest.mark.parametrize("src", ["corebench", "bixbench", "biomysterybench"])
    def test_source_info_records_ai_for_science_domain(self, src):
        # Arrange
        from scitex_dataset._sources import SOURCE_INFO

        # Act
        domain = SOURCE_INFO[src]["domain"]
        # Assert
        assert domain == "ai-for-science"

    @pytest.mark.parametrize("src", ["corebench", "bixbench", "biomysterybench"])
    def test_source_info_records_agentic_bench_kind(self, src):
        # Arrange
        from scitex_dataset._sources import SOURCE_INFO

        # Act
        kind = SOURCE_INFO[src]["kind"]
        # Assert
        assert kind == "agentic-bench"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
