#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/project/provenance.py

"""Provenance hook: which scripts READ a dataset — a versioned, neutral contract.

scitex-dataset does not know any provenance system. It publishes this
contract and asks whatever providers are installed; a provider (for example
scitex-clew's adapter) lives on the PROVIDER's side and registers itself in
the ``scitex.dataset.provenance`` entry-point group::

    [project.entry-points."scitex.dataset.provenance"]
    myprovider = "mypkg._dataset_readers:ReadersEntry"

Contract ``scitex-dataset.provenance-readers`` version :data:`CONTRACT_VERSION`
(the JSON Schema of the return value ships next to this module as
``provenance-readers.schema.<version>.json``). The entry-point object — a
class (instantiated with no arguments) or an instance — exposes:

``name: str``
    Shown to the user as the source of the records.
``contract_version: str``
    The contract version it implements. Only the MAJOR must match.
``readers(project_root: str, paths: list[str]) -> dict[str, list[dict]]``
    For each project-relative POSIX path in ``paths`` that the provider has
    seen READ, a list of records ``{"script": str, "run_id": str | None,
    "started_at": str | None, "status": str | None}``. ``script`` is
    project-relative when the script lives in the project. Paths it knows
    nothing about are simply absent. It must not write to the project.

A provider that raises, returns the wrong shape, or speaks another major
version is reported per provider and never breaks the listing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable, Optional

CONTRACT_NAME = "scitex-dataset.provenance-readers"
CONTRACT_VERSION = "1.0.0"
ENTRY_POINT_GROUP = "scitex.dataset.provenance"
SCHEMA_PATH = Path(__file__).with_name(
    f"provenance-readers.schema.{CONTRACT_VERSION}.json"
)

_RECORD_FIELDS = ("script", "run_id", "started_at", "status")
_MAX_RECORDS_PER_PATH = 50


def load_schema() -> dict:
    """The JSON Schema (draft 2020-12) of a provider's ``readers`` return value."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def discover_providers() -> list:
    """Installed providers, one per entry-point name, sorted by name."""
    from importlib.metadata import entry_points

    found = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        found.setdefault(ep.name, ep)
    return [found[name] for name in sorted(found)]


def _major(version: Any) -> Optional[str]:
    return version.split(".", 1)[0] if isinstance(version, str) and version else None


def _instantiate(obj):
    return obj() if isinstance(obj, type) else obj


def _clean_records(records: Any, provider: str) -> list[dict]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a list")
    out = []
    for record in list(records)[:_MAX_RECORDS_PER_PATH]:
        if not isinstance(record, Mapping):
            raise TypeError("each record must be an object")
        script = record.get("script")
        if not isinstance(script, str) or not script:
            raise TypeError("each record needs a non-empty 'script'")
        clean = {"provider": provider}
        for field in _RECORD_FIELDS:
            value = record.get(field)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"record field {field!r} must be a string or null")
            clean[field] = value
        out.append(clean)
    return out


def readers_of(root, paths: Iterable[str], providers: Optional[list] = None) -> dict:
    """Ask every provider which scripts read ``paths`` (project-relative).

    ``providers`` defaults to the installed entry points; pass loaded
    provider objects to use specific ones. Returns::

        {"contract": CONTRACT_NAME, "contract_version": CONTRACT_VERSION,
         "providers": [{"name", "status": "ok"|"error"|"incompatible", "error"}],
         "readers": {path: [{"provider", "script", "run_id", "started_at", "status"}]}}
    """
    paths = [p for p in paths if isinstance(p, str) and p]
    wanted = set(paths)
    result = {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "providers": [],
        "readers": {},
    }
    if providers is None:
        loaded = []
        for ep in discover_providers():
            try:
                loaded.append((ep.name, _instantiate(ep.load())))
            except Exception as exc:  # noqa: BLE001 - third-party provider boundary
                result["providers"].append(
                    {"name": ep.name, "status": "error", "error": f"cannot load: {exc}"}
                )
    else:
        loaded = [
            (getattr(p, "name", type(p).__name__), _instantiate(p)) for p in providers
        ]

    for fallback_name, provider in loaded:
        name = getattr(provider, "name", None) or fallback_name
        if _major(getattr(provider, "contract_version", None)) != _major(
            CONTRACT_VERSION
        ):
            result["providers"].append(
                {
                    "name": name,
                    "status": "incompatible",
                    "error": (
                        f"implements {getattr(provider, 'contract_version', None)!r}, "
                        f"this scitex-dataset speaks {CONTRACT_VERSION}"
                    ),
                }
            )
            continue
        if not paths:
            result["providers"].append({"name": name, "status": "ok", "error": None})
            continue
        try:
            answer = provider.readers(str(Path(root).resolve()), list(paths))
            if not isinstance(answer, Mapping):
                raise TypeError("readers() must return an object keyed by path")
            cleaned = {
                path: _clean_records(records, name)
                for path, records in answer.items()
                if path in wanted
            }
        except Exception as exc:  # noqa: BLE001 - third-party provider boundary
            result["providers"].append(
                {"name": name, "status": "error", "error": str(exc)}
            )
            continue
        for path, records in cleaned.items():
            if records:
                result["readers"].setdefault(path, []).extend(records)
        result["providers"].append({"name": name, "status": "ok", "error": None})
    return result


__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "ENTRY_POINT_GROUP",
    "SCHEMA_PATH",
    "discover_providers",
    "load_schema",
    "readers_of",
]

# EOF
