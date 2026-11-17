#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_materialization_identity.py

"""Operator-private cache identity ledger for per-capsule materialization.

Split from ``_standardize``. Existing caches are reused only when their
tasks/archive/source/masking/identifier-policy identity still matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from ._capsule_layout import (
    _CAPSULE_DIR_PREFIX,
    _IDENTITY_FILENAME,
    _INDEX_FILENAME,
    _OPERATOR_ROOT_FILES,
)
from ._identifiers import IDENTIFIER_POLICY, native_identifiers
from ._leak_strip import InvalidReferenceSourceError, _answer_targets

# v2: identity also binds the hosting-identifier redaction policy.
_MATERIALIZATION_CONTRACT = "per-capsule-source-identity-v2"


class StaleMaterializationError(ValueError):
    """Cached identity is unknown or changed; use a fresh output directory."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _materialized_identity(capsule_dir: Path) -> str:
    """Hash actual cached names/bytes/modes; never follow a symlink target."""
    entries = []
    for parent, directories, files in os.walk(capsule_dir, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            mode = path.lstat().st_mode
            row = {
                "path": path.relative_to(capsule_dir).as_posix(),
                "executable": mode & 0o111,
            }
            if stat.S_ISLNK(mode):
                row.update(kind="symlink", target=os.readlink(path))
            elif stat.S_ISREG(mode):
                row.update(kind="file", sha256=_sha256_file(path))
            elif stat.S_ISDIR(mode):
                row.update(kind="directory")
            else:
                raise StaleMaterializationError("unsupported cached filesystem entry")
            entries.append(row)
    return _json_identity(sorted(entries, key=lambda row: row["path"]))


def _capsule_reference_values(rows, answer_values, reference_values):
    if (answer_values is not None and not isinstance(answer_values, dict)) or (
        reference_values is not None and not isinstance(reference_values, dict)
    ):
        raise InvalidReferenceSourceError("reference mappings must be objects")
    values = [
        answer_values[row["task_id"]]
        for row in rows
        if answer_values and row["task_id"] in answer_values
    ]
    for row in rows:
        samples = (reference_values or {}).get(row["task_id"], [])
        if not isinstance(samples, list):
            raise InvalidReferenceSourceError(
                "reference_values requires lists of samples"
            )
        values.extend(samples)
    return values


def _capsule_identifiers(native, identifiers):
    """Exact hosting identifiers for one capsule: native id forms + adapter's."""
    if identifiers is not None and not isinstance(identifiers, dict):
        raise InvalidReferenceSourceError("identifiers must map native ids to lists")
    extra = (identifiers or {}).get(native, [])
    if not isinstance(extra, list) or not all(isinstance(v, str) for v in extra):
        raise InvalidReferenceSourceError("identifiers requires lists of strings")
    return sorted(set(native_identifiers(native) + extra))


def _qualify_materializations(
    for_solver_dir,
    index,
    selected,
    by_native,
    raw_dir,
    source_identity,
    answer_values,
    reference_values,
    identifiers=None,
):
    """Preflight ALL existing caches before mapper/purge/extraction effects.

    Only selected or already-materialized archives are read. The optional
    caller identity belongs to this operator-private root ledger, not the
    solver task/schema. Actual upstream revision qualification is separate.
    """
    ledger_path = for_solver_dir / _IDENTITY_FILENAME
    old = {"contract": _MATERIALIZATION_CONTRACT, "materializations": {}}
    existing = []
    if for_solver_dir.exists():
        if not for_solver_dir.is_dir() or for_solver_dir.is_symlink():
            raise StaleMaterializationError("output root must be a real directory")
        for entry in for_solver_dir.iterdir():
            if (
                entry.name in _OPERATOR_ROOT_FILES
                and entry.is_file()
                and not entry.is_symlink()
            ):
                continue
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and entry.name.startswith(_CAPSULE_DIR_PREFIX)
                and entry.name[len(_CAPSULE_DIR_PREFIX) :].isdigit()
            ):
                existing.append(entry.name)
            else:
                raise StaleMaterializationError(
                    "legacy root artifacts retained; use a fresh directory"
                )
        if any(for_solver_dir.iterdir()):
            if not ledger_path.is_file() or ledger_path.is_symlink():
                raise StaleMaterializationError(
                    "legacy cache has no qualified identity; retained unchanged"
                )
            try:
                old = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise StaleMaterializationError(
                    "invalid cache identity ledger"
                ) from exc
            if (
                not isinstance(old, dict)
                or old.get("contract") != _MATERIALIZATION_CONTRACT
                or not isinstance(old.get("materializations"), dict)
            ):
                raise StaleMaterializationError(
                    "cache contract changed; retained unchanged"
                )
            cached_index = for_solver_dir / _INDEX_FILENAME
            if not cached_index.is_file() or old.get("index_sha256") != _sha256_file(
                cached_index
            ):
                raise StaleMaterializationError(
                    "cached mapper changed; retained unchanged"
                )
    current = {row["dir"]: row for row in index}
    needed = set(existing) | {row["dir"] for row in selected}
    identities = {}
    for name in sorted(needed):
        row = current.get(name)
        if row is None:
            raise StaleMaterializationError(
                "existing capsule is absent from new mapping"
            )
        tasks = by_native[row["native_id"]]
        values = _capsule_reference_values(tasks, answer_values, reference_values)
        # Exercise the UNCHANGED target policy before extraction or mutation.
        # Invalid numeric source values refuse rather than being dropped.
        _answer_targets(values)
        try:
            masking_identity = _json_identity(values)
        except (TypeError, ValueError) as exc:
            raise InvalidReferenceSourceError(
                "masking references must be valid JSON"
            ) from exc
        archive_paths = {task["data"] for task in tasks}
        if len(archive_paths) != 1:
            raise ValueError("a native capsule must have exactly one archive identity")
        # Check mapping before touching the archive for a new/different native.
        previous = old.get("materializations", {}).get(name)
        if name in existing and (
            not isinstance(previous, dict)
            or previous.get("native_id") != row["native_id"]
        ):
            raise StaleMaterializationError(
                "friendly/native mapping changed; retained unchanged"
            )
        archive_path = (raw_dir / next(iter(archive_paths))).resolve()
        identity = {
            "native_id": row["native_id"],
            "tasks_sha256": _json_identity(tasks),
            "archive_sha256": _sha256_file(archive_path),
            "source_identity_sha256": _json_identity(source_identity),
            "masking_values_sha256": masking_identity,
            "identifier_policy_sha256": _json_identity(
                {
                    "policy": IDENTIFIER_POLICY,
                    "exact": _capsule_identifiers(row["native_id"], identifiers),
                }
            ),
        }
        if name in existing:
            if any(previous.get(key) != value for key, value in identity.items()):
                raise StaleMaterializationError(
                    "task/archive/source identity changed; retained unchanged"
                )
            if previous.get("materialized_sha256") != _materialized_identity(
                for_solver_dir / name
            ):
                raise StaleMaterializationError(
                    "cached materialization changed; retained unchanged"
                )
        identities[name] = identity
    return old, identities


# EOF
