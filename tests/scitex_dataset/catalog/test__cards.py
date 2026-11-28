#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/catalog/test__cards.py

"""Cards: one shape for every repository, never inventing a fact."""

from __future__ import annotations

import pytest

from scitex_dataset.catalog import (
    card_from_record,
    get_dataset,
    human_size,
    search_public,
    split_key,
)


@pytest.mark.parametrize(
    ("n_bytes", "text"),
    [(0, "0 B"), (999, "999 B"), (11_490_434, "11.5 MB"), (2_500_000_000, "2.5 GB")],
)
def test_human_size_uses_decimal_units(n_bytes, text):
    # Arrange
    value = n_bytes
    # Act
    shown = human_size(value)
    # Assert
    assert shown == text


def test_human_size_of_unknown_stays_unknown():
    # Arrange
    value = None
    # Act
    shown = human_size(value)
    # Assert
    assert shown is None


def test_an_absent_licence_is_not_invented():
    # Arrange
    record = {"id": "ds1", "name": "A"}
    # Act
    card = card_from_record(record, "openneuro")
    # Assert
    assert card["license"] is None


def test_a_doi_is_extracted_from_a_doi_url():
    # Arrange
    record = {"id": "ds1", "doi": "https://doi.org/10.18112/openneuro.ds1.v1.0.0"}
    # Act
    card = card_from_record(record, "openneuro")
    # Assert
    assert card["doi"] == "10.18112/openneuro.ds1.v1.0.0"


def test_size_in_gib_becomes_bytes():
    # Arrange
    record = {"id": "ds1", "size_gb": 2}
    # Act
    card = card_from_record(record, "physionet")
    # Assert
    assert card["size_bytes"] == 2 * 1024**3


def test_the_repository_acknowledgement_is_the_citation():
    # Arrange
    record = {"id": "ds1", "how_to_acknowledge": "Please cite Smith 2020."}
    # Act
    card = card_from_record(record, "openneuro")
    # Assert
    assert card["citation"] == "Please cite Smith 2020."


def test_a_citation_is_built_from_facts_when_none_is_given():
    # Arrange
    record = {
        "id": "ds1",
        "name": "Motor imagery",
        "authors": ["A. B"],
        "created": "2021-03-01",
        "doi": "10.1234/x",
    }
    # Act
    card = card_from_record(record, "openneuro")
    # Assert
    assert (
        card["citation"]
        == "A. B (2021) Motor imagery. OpenNeuro. https://doi.org/10.1234/x"
    )


def test_catalogue_only_snippet_says_it_cannot_download():
    # Arrange
    record = {"id": "ds1"}
    # Act
    card = card_from_record(record, "openneuro")
    # Assert
    assert "not yet downloadable" in card["snippet"]


def test_split_key_refuses_a_bare_word():
    # Arrange
    key = "mnist"
    # Act
    try:
        split_key(key)
        raised = False
    except ValueError:
        raised = True
    # Assert
    assert raised


def test_mnist_idx_reference_card_has_four_files():
    # Arrange
    key = "reference:mnist-idx"
    # Act
    card = get_dataset(key)
    # Assert
    assert len(card["files"]) == 4


def test_mnist_idx_reference_version_names_the_pinned_bytes():
    # Arrange
    key = "reference:mnist-idx"
    # Act
    card = get_dataset(key)
    # Assert
    assert card["version"].startswith("sha256:")


def test_reference_card_does_not_claim_the_paper_doi_for_the_data():
    # Arrange
    key = "reference:mnist"
    # Act
    card = get_dataset(key)
    # Assert
    assert card["doi"] is None


def test_reference_search_works_without_the_index():
    # Arrange
    query = "mnist idx"
    # Act
    found = search_public(query, repository="reference")
    # Assert
    assert [c["key"] for c in found["results"]] == ["reference:mnist-idx"]


def test_an_unknown_repository_is_refused():
    # Arrange
    repository = "nowhere"
    # Act
    try:
        search_public("x", repository=repository)
        raised = False
    except ValueError:
        raised = True
    # Assert
    assert raised


# EOF
