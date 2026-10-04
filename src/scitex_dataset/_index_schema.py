#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_index_schema.py

"""What a dataset record IS — the domain half of the index.

Separated from :mod:`scitex_dataset.database` because the two change for
different reasons and one of them is dangerous to change casually. This
file is the SHAPE: which columns a dataset has, how two concurrent values
for one of them reconcile, which are worth an index and which are worth
searching. ``database.py`` is the VERBS: build, search, stats, clear.

There is no field default here and there is not meant to be one. The store
primitive refuses to invent a merge rule, and the refusal is the feature: a
wrong default loses data with nothing raised, days before anyone notices
that "the data is wrong". So every column below states its own policy, even
where the answer looks obvious.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from scitex_dev.store import (
    FieldKind,
    FieldPolicy,
    FieldRole,
    MergeRule,
    Schema,
)

__all__ = ["PKG", "SCHEMA", "VALID_ORDERS", "row_values"]

#: The package short name the store resolves under.
PKG = "dataset"

#: Sort keys :func:`scitex_dataset.database.search` accepts. Anything else
#: falls back to ``downloads``, as it always has — an unknown sort key is a
#: caller's typo, not a reason to refuse to answer.
VALID_ORDERS = ("downloads", "views", "n_subjects", "size_gb", "name", "created")


def _text(*, indexed: bool = False, required: bool = False) -> FieldPolicy:
    return FieldPolicy(
        kind=FieldKind.TEXT,
        role=FieldRole.DATA,
        required=required,
        merge=MergeRule.LAST_WRITER_WINS,
        indexed=indexed,
    )


def _number(kind: FieldKind, *, indexed: bool = False) -> FieldPolicy:
    return FieldPolicy(
        kind=kind,
        role=FieldRole.DATA,
        required=False,
        merge=MergeRule.LAST_WRITER_WINS,
        indexed=indexed,
    )


def _json() -> FieldPolicy:
    return FieldPolicy(
        kind=FieldKind.JSON,
        role=FieldRole.DATA,
        required=False,
        merge=MergeRule.LAST_WRITER_WINS,
        indexed=False,
    )


#: One dataset record.
#:
#: ``id`` is IMMUTABLE because it IS the record: changing it does not rename
#: a dataset, it names a different one. It is namespaced ``<source>:<id>``,
#: because two catalogues numbering their datasets from one would otherwise
#: collide on the key and each rebuild would overwrite the other's.
#:
#: ``record`` keeps the fetcher's own dictionary verbatim. The flat columns
#: exist so the store can filter and sort on them; this is what a caller
#: gets back, so a field a fetcher adds upstream reaches users without a
#: schema change here.
#:
#: ``retired`` is the soft-delete flag. Nothing is ever deleted; a retired
#: dataset leaves the default view and stays readable.
#:
#: ``text_search`` names the same four columns the previous full-text index
#: covered, so a query that matched before matches now. It is declared once
#: and the store builds BOTH its index and its match expression from this
#: single list — two copies would eventually differ, and an expression index
#: that differs from its query is silently never used.
SCHEMA: Schema = Schema.build(
    "datasets",
    {
        "id": FieldPolicy(
            kind=FieldKind.TEXT,
            role=FieldRole.IDENTITY,
            required=True,
            merge=MergeRule.IMMUTABLE,
            indexed=False,
        ),
        "source": _text(indexed=True, required=True),
        "name": _text(),
        "created": _text(),
        "modified": _text(),
        "n_subjects": _number(FieldKind.INTEGER, indexed=True),
        "size_gb": _number(FieldKind.REAL),
        "downloads": _number(FieldKind.INTEGER, indexed=True),
        "views": _number(FieldKind.INTEGER),
        "readme": _text(),
        "license": _text(),
        "doi": _text(),
        "url": _text(),
        "modalities": _json(),
        "tasks": _json(),
        "primary_modality": _text(),
        "record": _json(),
        "indexed_at": _text(indexed=True),
        "retired": FieldPolicy(
            kind=FieldKind.BOOL,
            role=FieldRole.HIDE_FLAG,
            required=False,
            merge=MergeRule.LAST_WRITER_WINS,
            indexed=False,
        ),
    },
    text_search=("id", "name", "readme", "tasks"),
)


def row_values(dataset: Dict[str, Any], source: str) -> Dict[str, Any]:
    """One fetched dataset, shaped for :data:`SCHEMA`.

    The numeric defaults are zeros rather than ``None`` because the sort
    keys are built on them: a dataset with no recorded download count
    should rank at the bottom of "most downloaded", not vanish from it.
    """
    return {
        "id": f"{source}:{dataset.get('id', '')}",
        "source": source,
        "name": dataset.get("name"),
        "created": dataset.get("created"),
        "modified": dataset.get("modified"),
        "n_subjects": dataset.get("n_subjects", 0),
        "size_gb": dataset.get("size_gb", 0),
        "downloads": dataset.get("downloads", 0),
        "views": dataset.get("views", 0),
        "readme": dataset.get("readme"),
        "license": dataset.get("license"),
        "doi": dataset.get("doi"),
        "url": dataset.get("url"),
        "modalities": dataset.get("modalities", []),
        "tasks": dataset.get("tasks", []),
        "primary_modality": dataset.get("primary_modality"),
        "record": dataset,
        "indexed_at": datetime.now().isoformat(),
        "retired": False,
    }

# EOF
