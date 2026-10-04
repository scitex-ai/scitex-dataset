#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/test_database.py

"""The dataset index, against the store that actually holds it.

EVERY TEST HERE USES A REAL POSTGRESQL SCHEMA, created before it and
dropped after (see the ``index_store`` fixture). There is no in-memory
double and no patched connection, because the questions being asked are
about behaviour the engine decides: whether a rebuild replaces a row rather
than duplicating it, whether ``NULLS LAST`` keeps unmeasured downloads out
of the top of "most downloaded", whether JSON containment means containment.

The previous version of this file drove a private database file directly —
``_get_connection``, ``_insert_dataset``, raw SQL — and asserted on tables
this package owned. It owns none now: the shape is a schema declaration
handed to the primitive, so what is left to test is the DOMAIN. That is why
these tests go through ``database.build``/``search``/``get_stats`` rather
than through anything below them.
"""

from __future__ import annotations

import pytest

from scitex_dataset import database
from scitex_dataset._index_schema import SCHEMA, row_values


def _seed(store, records) -> None:
    """Write ``(dataset, source)`` pairs the way ``build`` would."""
    from scitex_dev.store import ANY_REVISION

    for dataset, source in records:
        store.put(row_values(dataset, source), expected_revision=ANY_REVISION)


@pytest.fixture
def two_sources(index_store):
    """Three datasets: two from openneuro, one from dandi."""
    _seed(
        index_store,
        [
            ({"id": "ds000", "name": "Dataset 0"}, "openneuro"),
            ({"id": "ds001", "name": "Dataset 1"}, "openneuro"),
            ({"id": "ds002", "name": "Dataset 2"}, "dandi"),
        ],
    )
    return index_store


@pytest.fixture
def by_modality(index_store):
    """One MRI dataset, one EEG, one carrying both."""
    _seed(
        index_store,
        [
            ({"id": "ds001", "name": "MRI Study", "modalities": ["mri"]}, "openneuro"),
            ({"id": "ds002", "name": "EEG Study", "modalities": ["eeg"]}, "openneuro"),
            (
                {"id": "ds003", "name": "Multimodal", "modalities": ["mri", "eeg"]},
                "openneuro",
            ),
        ],
    )
    return index_store


@pytest.fixture
def by_subjects(index_store):
    """Three datasets with 10, 30 and 50 subjects."""
    _seed(
        index_store,
        [
            (
                {"id": f"ds{n}", "name": f"Study {n}", "n_subjects": n},
                "openneuro",
            )
            for n in (10, 30, 50)
        ],
    )
    return index_store


@pytest.fixture
def by_text(index_store):
    """One dataset about memory, one about movement."""
    _seed(
        index_store,
        [
            (
                {
                    "id": "ds001",
                    "name": "Alzheimer Study",
                    "readme": "Memory impairment",
                },
                "openneuro",
            ),
            (
                {
                    "id": "ds002",
                    "name": "Motor Control",
                    "readme": "Movement analysis",
                },
                "openneuro",
            ),
        ],
    )
    return index_store


@pytest.fixture
def ten_datasets(index_store):
    """Ten datasets with strictly decreasing download counts."""
    _seed(
        index_store,
        [
            (
                {"id": f"ds{i:03d}", "name": f"Dataset {i}", "downloads": 100 - i},
                "openneuro",
            )
            for i in range(10)
        ],
    )
    return index_store


# -- the schema ------------------------------------------------------------
def test_the_index_is_keyed_by_source_namespaced_id():
    """Two catalogues numbering from one would otherwise collide, and each
    rebuild would overwrite the other's rows."""
    # Arrange
    dataset = {"id": "ds001"}

    # Act
    values = row_values(dataset, "openneuro")

    # Assert
    assert values["id"] == "openneuro:ds001"


def test_the_index_keeps_the_fetchers_record_verbatim():
    """So a field a fetcher adds upstream reaches callers with no schema
    change here."""
    # Arrange
    dataset = {"id": "ds001", "surprise": "a field this schema never named"}

    # Act
    values = row_values(dataset, "openneuro")

    # Assert
    assert values["record"] == dataset


def test_store_description_names_the_store_it_was_given(index_store):
    """The CLI prints this where it used to print a path. It must describe
    the store actually in use, not the one this host would resolve."""
    # Arrange
    store = index_store

    # Act
    described = database.store_description(store=store)

    # Assert
    assert described == store.target.describe()


def test_the_schema_declares_its_searchable_fields():
    # Arrange
    expected = ("id", "name", "readme", "tasks")

    # Act
    declared = SCHEMA.text_search

    # Assert
    assert declared == expected


# -- writing ---------------------------------------------------------------
def test_indexing_a_dataset_makes_it_findable(index_store):
    # Arrange
    _seed(index_store, [({"id": "ds001", "name": "Test Dataset"}, "openneuro")])

    # Act
    found = database.search(store=index_store)

    # Assert
    assert len(found) == 1


def test_indexing_a_dataset_preserves_its_name(index_store):
    # Arrange
    _seed(index_store, [({"id": "ds001", "name": "Test Dataset"}, "openneuro")])

    # Act
    found = database.search(store=index_store)

    # Assert
    assert found[0]["name"] == "Test Dataset"


def test_reindexing_the_same_dataset_keeps_one_row(index_store):
    """A rebuild must refresh, not duplicate."""
    # Arrange
    _seed(index_store, [({"id": "ds001", "name": "Original"}, "openneuro")])

    # Act
    _seed(index_store, [({"id": "ds001", "name": "Updated"}, "openneuro")])

    # Assert
    assert index_store.count() == 1


def test_reindexing_the_same_dataset_overwrites_its_name(index_store):
    # Arrange
    _seed(index_store, [({"id": "ds001", "name": "Original"}, "openneuro")])

    # Act
    _seed(index_store, [({"id": "ds001", "name": "Updated"}, "openneuro")])

    # Assert
    assert database.search(store=index_store)[0]["name"] == "Updated"


def test_two_sources_numbering_from_one_do_not_collide(index_store):
    """The namespaced key, proven rather than asserted about a helper."""
    # Arrange
    _seed(index_store, [({"id": "ds001"}, "openneuro")])

    # Act
    _seed(index_store, [({"id": "ds001"}, "dandi")])

    # Assert
    assert index_store.count() == 2


# -- searching -------------------------------------------------------------
def test_search_of_an_empty_index_returns_an_empty_list(index_store):
    # Arrange
    expected = []

    # Act
    results = database.search(store=index_store)

    # Assert
    assert results == expected


def test_search_by_source_openneuro_returns_two_rows(two_sources):
    # Arrange
    source = "openneuro"

    # Act
    results = database.search(source=source, store=two_sources)

    # Assert
    assert len(results) == 2


def test_search_by_source_dandi_returns_one_row(two_sources):
    # Arrange
    source = "dandi"

    # Act
    results = database.search(source=source, store=two_sources)

    # Assert
    assert len(results) == 1


def test_search_by_modality_eeg_returns_two_rows(by_modality):
    # Arrange
    modality = "eeg"

    # Act
    results = database.search(modality=modality, store=by_modality)

    # Assert
    assert len(results) == 2


def test_search_by_modality_asks_the_list_not_the_prose(index_store):
    """The old filter matched a substring of the serialised column, so a
    dataset whose free text said "eeg" could answer a modality query."""
    # Arrange
    _seed(
        index_store,
        [
            (
                {
                    "id": "ds001",
                    "name": "MRI only",
                    "readme": "compared against eeg elsewhere",
                    "modalities": ["mri"],
                },
                "openneuro",
            )
        ],
    )

    # Act
    results = database.search(modality="eeg", store=index_store)

    # Assert
    assert results == []


def test_search_by_modality_matches_the_primary_modality(index_store):
    """One user-facing filter spanning two columns."""
    # Arrange
    _seed(
        index_store,
        [({"id": "ds001", "primary_modality": "eeg"}, "openneuro")],
    )

    # Act
    results = database.search(modality="eeg", store=index_store)

    # Assert
    assert len(results) == 1


def test_search_by_min_subjects_returns_only_high_count_rows(by_subjects):
    # Arrange
    minimum = 25

    # Act
    results = database.search(min_subjects=minimum, store=by_subjects)

    # Assert
    assert len(results) == 2


def test_search_by_max_subjects_returns_only_low_count_rows(by_subjects):
    # Arrange
    maximum = 35

    # Act
    results = database.search(max_subjects=maximum, store=by_subjects)

    # Assert
    assert len(results) == 2


def test_search_by_min_and_max_subjects_returns_one_row(by_subjects):
    # Arrange
    bounds = (25, 35)

    # Act
    results = database.search(
        min_subjects=bounds[0], max_subjects=bounds[1], store=by_subjects
    )

    # Assert
    assert len(results) == 1


def test_search_by_min_downloads_filters_out_the_quiet_ones(ten_datasets):
    # Arrange
    minimum = 95

    # Act
    results = database.search(min_downloads=minimum, store=ten_datasets)

    # Assert
    assert len(results) == 6


def test_has_readme_rejects_an_empty_readme(index_store):
    """`IS NOT NULL` alone would accept it, and "has a readme" would mean
    "has the column"."""
    # Arrange
    _seed(index_store, [({"id": "ds001", "readme": ""}, "openneuro")])

    # Act
    results = database.search(has_readme=True, store=index_store)

    # Assert
    assert results == []


def test_search_full_text_query_returns_one_match(by_text):
    # Arrange
    query = "alzheimer"

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert len(results) == 1


def test_search_full_text_query_returns_matching_id(by_text):
    # Arrange
    query = "alzheimer"

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert results[0]["id"] == "ds001"


def test_search_full_text_reaches_the_readme(by_text):
    # Arrange
    query = "movement"

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert results[0]["id"] == "ds002"


def test_search_full_text_ands_bare_words(by_text):
    # Arrange
    query = "motor movement"

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert len(results) == 1


def test_search_full_text_matching_nothing_returns_nothing(by_text):
    """The control for the four above: they would look identical if the
    match expression accepted every row."""
    # Arrange
    query = "thermodynamics"

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert results == []


def test_search_full_text_survives_a_half_typed_query(by_text):
    """A search box must not be able to raise."""
    # Arrange
    query = '"alzheimer'

    # Act
    results = database.search(query=query, store=by_text)

    # Assert
    assert isinstance(results, list)


def test_search_with_limit_returns_at_most_three_rows(ten_datasets):
    # Arrange
    limit = 3

    # Act
    results = database.search(limit=limit, store=ten_datasets)

    # Assert
    assert len(results) == 3


def test_search_with_offset_returns_next_page_of_three_rows(ten_datasets):
    # Arrange
    limit, offset = 3, 3

    # Act
    results = database.search(limit=limit, offset=offset, store=ten_datasets)

    # Assert
    assert len(results) == 3


def test_search_pages_do_not_overlap(ten_datasets):
    # Arrange
    first = database.search(limit=3, store=ten_datasets)

    # Act
    second = database.search(limit=3, offset=3, store=ten_datasets)

    # Assert
    assert {d["id"] for d in first}.isdisjoint({d["id"] for d in second})


def test_search_order_by_downloads_returns_descending(index_store):
    # Arrange
    _seed(
        index_store,
        [
            (
                {"id": f"ds{i:03d}", "downloads": (i + 1) * 100, "n_subjects": 30 - i * 10},
                "openneuro",
            )
            for i in range(3)
        ],
    )

    # Act
    results = database.search(order_by="downloads", store=index_store)

    # Assert
    assert results[0]["downloads"] > results[-1]["downloads"]


def test_search_order_by_n_subjects_returns_descending(index_store):
    # Arrange
    _seed(
        index_store,
        [
            (
                {"id": f"ds{i:03d}", "downloads": (i + 1) * 100, "n_subjects": 30 - i * 10},
                "openneuro",
            )
            for i in range(3)
        ],
    )

    # Act
    results = database.search(order_by="n_subjects", store=index_store)

    # Assert
    assert results[0]["n_subjects"] > results[-1]["n_subjects"]


def test_an_unknown_sort_key_falls_back_to_downloads(ten_datasets):
    """A caller's typo is not a reason to refuse to answer."""
    # Arrange
    order_by = "populariy"

    # Act
    results = database.search(order_by=order_by, store=ten_datasets)

    # Assert
    assert results[0]["downloads"] == 100


# -- statistics ------------------------------------------------------------
def test_get_stats_reports_not_built_when_the_index_is_empty(index_store):
    # Arrange
    store = index_store

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["exists"] is False


def test_get_stats_says_how_to_build_when_the_index_is_empty(index_store):
    # Arrange
    store = index_store

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert "not built" in stats["message"].lower()


def test_get_stats_names_the_store_even_when_empty(index_store):
    """"Where would it have gone?" is the first question after an empty
    answer, and it used to be unanswerable without a second call."""
    # Arrange
    store = index_store

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert "postgres" in stats["store"]


def test_get_stats_reports_exists_true_once_seeded(two_sources):
    # Arrange
    store = two_sources

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["exists"] is True


def test_get_stats_reports_the_total(two_sources):
    # Arrange
    store = two_sources

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["total_datasets"] == 3


def test_get_stats_by_source_counts_openneuro(two_sources):
    # Arrange
    store = two_sources

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["by_source"]["openneuro"] == 2


def test_get_stats_by_source_counts_dandi(two_sources):
    # Arrange
    store = two_sources

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["by_source"]["dandi"] == 1


def test_get_stats_reports_when_the_index_was_last_written(two_sources):
    # Arrange
    store = two_sources

    # Act
    stats = database.get_stats(store=store)

    # Assert
    assert stats["last_build"] is not None


# -- clearing --------------------------------------------------------------
def test_clear_empties_the_default_view(two_sources):
    # Arrange
    store = two_sources

    # Act
    database.clear(store=store)

    # Assert
    assert database.search(store=store) == []


def test_clear_returns_true_when_something_was_indexed(two_sources):
    # Arrange
    store = two_sources

    # Act
    result = database.clear(store=store)

    # Assert
    assert result is True


def test_clear_returns_false_on_an_already_empty_index(two_sources):
    # Arrange
    database.clear(store=two_sources)

    # Act
    result = database.clear(store=two_sources)

    # Assert
    assert result is False


def test_clear_does_not_delete_anything(two_sources):
    """The whole reason `clear` is a hide: a row that leaves the view must
    still be there. Nothing in this store can remove one."""
    # Arrange
    from scitex_dev.store import Query

    # Act
    database.clear(store=two_sources)

    # Assert
    assert len(two_sources.search(Query().with_hidden())) == 3


def test_rebuilding_after_clear_brings_a_dataset_back(two_sources):
    # Arrange
    database.clear(store=two_sources)

    # Act
    _seed(two_sources, [({"id": "ds000", "name": "Dataset 0"}, "openneuro")])

    # Assert
    assert len(database.search(store=two_sources)) == 1

# EOF
