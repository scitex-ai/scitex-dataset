#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/project/test_provenance.py

"""The provider-neutral provenance-readers contract, with real provider objects."""

from __future__ import annotations

import jsonschema

from scitex_dataset.project.provenance import (
    CONTRACT_VERSION,
    ENTRY_POINT_GROUP,
    load_schema,
    readers_of,
)

RECORD = {
    "script": "scripts/load.py",
    "run_id": "r1",
    "started_at": "2026-11-26T00:00:00",
    "status": "success",
}


class StaticReaders:
    """A provider answering from a fixed table (the contract's simplest implementer)."""

    name = "static"
    contract_version = CONTRACT_VERSION

    def __init__(self, table=None):
        self.table = {"data/a.csv": [RECORD]} if table is None else table
        self.asked = []

    def readers(self, project_root, paths):
        self.asked.append((project_root, list(paths)))
        return {p: v for p, v in self.table.items() if p in paths}


class RaisingReaders(StaticReaders):
    name = "raising"

    def readers(self, project_root, paths):
        raise RuntimeError("store offline")


class FutureReaders(StaticReaders):
    name = "future"
    contract_version = "2.0.0"


class MalformedReaders(StaticReaders):
    name = "malformed"

    def readers(self, project_root, paths):
        return {"data/a.csv": [{"script": 7}]}


def test_records_are_returned_per_path_with_their_provider(tmp_path):
    # Arrange
    provider = StaticReaders()
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[provider])
    # Assert
    assert result["readers"] == {"data/a.csv": [{"provider": "static", **RECORD}]}


def test_the_result_names_the_contract_and_version(tmp_path):
    # Arrange
    provider = StaticReaders()
    # Act
    result = readers_of(tmp_path, [], providers=[provider])
    # Assert
    assert (result["contract"], result["contract_version"]) == (
        "scitex-dataset.provenance-readers",
        CONTRACT_VERSION,
    )


def test_paths_not_asked_for_are_dropped(tmp_path):
    # Arrange
    provider = StaticReaders({"data/a.csv": [RECORD], "data/other.csv": [RECORD]})
    provider.readers = lambda root, paths: provider.table  # answers more than asked
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[provider])
    # Assert
    assert list(result["readers"]) == ["data/a.csv"]


def test_a_raising_provider_is_reported_not_fatal(tmp_path):
    # Arrange
    providers = [RaisingReaders(), StaticReaders()]
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=providers)
    # Assert
    assert [(p["name"], p["status"]) for p in result["providers"]] == [
        ("raising", "error"),
        ("static", "ok"),
    ]


def test_another_major_version_is_incompatible_and_not_called(tmp_path):
    # Arrange
    provider = FutureReaders()
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[provider])
    # Assert
    assert (result["providers"][0]["status"], provider.asked) == ("incompatible", [])


def test_a_malformed_answer_is_an_error_and_contributes_nothing(tmp_path):
    # Arrange
    provider = MalformedReaders()
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[provider])
    # Assert
    assert (result["providers"][0]["status"], result["readers"]) == ("error", {})


def test_no_provider_means_an_empty_answer(tmp_path):
    # Arrange
    providers = []
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=providers)
    # Assert
    assert (result["providers"], result["readers"]) == ([], {})


def test_the_entry_point_group_is_the_published_name():
    # Arrange
    expected = "scitex.dataset.provenance"
    # Act
    group = ENTRY_POINT_GROUP
    # Assert
    assert group == expected


def test_a_conforming_answer_validates_against_the_shipped_schema():
    # Arrange
    schema = load_schema()
    # Act
    answer = StaticReaders().readers("/p", ["data/a.csv"])
    # Assert
    assert jsonschema.Draft202012Validator(schema).is_valid(answer)


def test_the_schema_rejects_a_record_without_a_script():
    # Arrange
    schema = load_schema()
    # Act
    answer = {"data/a.csv": [{"run_id": "r1"}]}
    # Assert
    assert not jsonschema.Draft202012Validator(schema).is_valid(answer)


# EOF


# --- contract 1.1: store routing and dataset sources --------------------------

ROUTE = {"tenant_id": "t", "dsn": "postgresql://h/db", "project_scope": "s", "owner_role": "o"}


class RoutedReaders(StaticReaders):
    """A 1.1 provider: records the store routing it was handed."""

    name = "routed"
    contract_version = "1.1.0"

    def readers(self, project_root, paths, *, store=None):
        self.asked.append(store)
        return {p: v for p, v in self.table.items() if p in paths}


class LegacyReaders(StaticReaders):
    """A 1.0 provider: its readers() takes no store keyword."""

    name = "legacy"
    contract_version = "1.0.0"


def test_a_1_1_provider_receives_the_store_routing(tmp_path):
    # Arrange
    provider = RoutedReaders()
    # Act
    readers_of(tmp_path, ["data/a.csv"], providers=[provider], store=ROUTE)
    # Assert
    assert provider.asked == [ROUTE]


def test_a_1_0_provider_is_called_without_the_routing(tmp_path):
    # Arrange
    provider = LegacyReaders()
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[provider], store=ROUTE)
    # Assert
    assert (result["providers"][0]["status"], result["providers"][0]["store_routed"]) == ("ok", False)


def test_the_result_names_the_public_dataset_of_each_path(tmp_path):
    # Arrange
    source = {"key": "reference:x", "repository": "reference", "accession": "x",
              "version": "v1", "doi": "10.1/x", "sha256": "ab" * 32}
    # Act
    result = readers_of(tmp_path, ["data/a.csv"], providers=[], sources={"data/a.csv": source})
    # Assert
    assert result["sources"]["data/a.csv"]["doi"] == "10.1/x"
