#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_neutral_paths.py

"""Neutral path names for the agent-visible ``for_solver/`` view (#111).

Content redaction (:mod:`._identifiers`, :mod:`._leak_strip`) rewrites file
CONTENTS. An upstream archive can also carry a source-benchmark identifier in
its file and directory NAMES (e.g. a top-level folder and notebooks named
after the capsule UUID), which a solver sees in any directory listing. This
module closes that channel:

1. :func:`neutralize_identifier_paths` renames every path component whose
   name carries an identifier to a neutral, deterministic name
   (``folder_01/``, ``notebook_01.ipynb``, ``file_01.csv``; extensions are
   kept), rewrites references to the old names inside every text file and
   symlink target so the project still runs, and returns the old→new
   mapping for the caller to record OUTSIDE ``for_solver/``.
2. :func:`assert_no_identifier_paths` is the fail-loud backstop: it refuses
   a tree in which any path component still carries an identifier.

A path component carries an identifier when (:func:`path_identifier_hits`)
it contains a UUID (any UUID: a name has no reason to embed one), any match
of the content matchers (:func:`._identifiers.find_identifiers`: hosting
URLs, DOIs, exact identifiers), an exact identifier delimited by any
non-alphanumeric character (so ``_`` and ``.`` count as separators in names),
a long digit core of an exact identifier (``capsule-7654321`` → ``7654321``),
or an adapter-supplied benchmark id pattern (e.g. BixBench ``bix-\\d+``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from ._identifiers import (
    _UUID_RE,
    _decode,
    _iter_text_files,
    _rewrite,
    find_identifiers,
)

# Bump when the rename or matching rules change (bound into cache identity).
PATH_POLICY = "neutral-paths-v1"

_MIN_EXACT_LEN = 6
_MIN_DIGIT_CORE = 6
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_NOTEBOOK_EXT = ".ipynb"


class IdentifierPathLeakError(RuntimeError):
    """A source identifier survived in a for_solver path component."""


def _delimited(value: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.IGNORECASE
    )


def _path_patterns(
    exact: Iterable[str], patterns: Iterable[str] = ()
) -> list[tuple[str, re.Pattern[str]]]:
    """Name-level matchers: exact ids, their digit cores, adapter patterns."""
    compiled: list[tuple[str, re.Pattern[str]]] = [("uuid", _UUID_RE)]
    values = {v.strip() for v in exact if isinstance(v, str)}
    for value in sorted(values, key=len, reverse=True):
        if len(value) >= _MIN_EXACT_LEN:
            compiled.append(("exact_identifier", _delimited(value)))
        for core in re.findall(r"\d+", value):
            if len(core) >= _MIN_DIGIT_CORE:
                compiled.append(
                    ("identifier_digits", re.compile(r"(?<!\d)" + core + r"(?!\d)"))
                )
    for pattern in patterns:
        compiled.append(("benchmark_id", re.compile(pattern, re.IGNORECASE)))
    return compiled


def path_identifier_hits(
    name: str, exact: Iterable[str] = (), patterns: Iterable[str] = ()
) -> list[tuple[str, str]]:
    """List every ``(class, matched text)`` identifier in one path component."""
    exact = list(exact)
    hits = list(find_identifiers(name, exact))
    for cls, pattern in _path_patterns(exact, patterns):
        hits.extend((cls, m.group(0)) for m in pattern.finditer(name))
    return hits


def _split_ext(name: str, is_dir: bool) -> tuple[str, str]:
    """``(stem, kept extension)``; up to two short alnum suffixes, never a dir's."""
    if is_dir:
        return name, ""
    stem, ext = name, ""
    for _ in range(2):
        match = _EXT_RE.search(stem)
        if not match or match.start() == 0:
            break
        ext = match.group(0) + ext
        stem = stem[: match.start()]
    return stem, ext


def _walk_names(root: Path):
    """Yield ``(relative path, is_dir)`` for every entry; never follow links."""
    for parent, dirs, files in os.walk(root, followlinks=False):
        base = Path(parent).relative_to(root)
        for name in sorted(dirs + files):
            path = Path(parent) / name
            yield base / name, path.is_dir() and not path.is_symlink()


def _sub_stems(text: str, regex: re.Pattern[str] | None, stems: dict) -> str:
    if regex is None:
        return text
    return regex.sub(lambda m: stems[m.group(0)], text)


def neutralize_identifier_paths(
    root: Path, exact: Iterable[str] = (), patterns: Iterable[str] = ()
) -> list[dict]:
    """Rename identifier-bearing path components under ``root``; keep refs.

    Each DISTINCT identifying stem gets one neutral stem — ``notebook_NN``
    when any entry with that stem is a notebook, else ``folder_NN`` for a
    directory, else ``file_NN`` — numbered per kind over the sorted stems,
    skipping numbers whose name already exists in the tree. The same stem
    therefore maps to the same neutral stem everywhere, so references stay
    consistent. Every text file under ``root`` and every symlink target then
    has the old stems (longest first, case-sensitive, delimited by
    non-alphanumerics) replaced by the neutral stems. Returns
    ``[{"old": rel, "new": rel}]`` (posix, sorted) for every renamed entry.
    """
    exact = list(exact)
    patterns = list(patterns)
    entries = [
        (rel, is_dir)
        for rel, is_dir in _walk_names(root)
        if path_identifier_hits(rel.name, exact, patterns)
    ]
    if not entries:
        return []
    taken = {
        _split_ext(rel.name, is_dir)[0].lower() for rel, is_dir in _walk_names(root)
    }
    kinds: dict[str, str] = {}
    order = {"notebook": 0, "folder": 1, "file": 2}
    for rel, is_dir in entries:
        stem, ext = _split_ext(rel.name, is_dir)
        kind = (
            "notebook"
            if ext.lower() == _NOTEBOOK_EXT
            else ("folder" if is_dir else "file")
        )
        if stem not in kinds or order[kind] < order[kinds[stem]]:
            kinds[stem] = kind
    stems: dict[str, str] = {}
    counters = {kind: 0 for kind in order}
    for stem in sorted(kinds):
        kind = kinds[stem]
        while True:
            counters[kind] += 1
            candidate = f"{kind}_{counters[kind]:02d}"
            if candidate not in taken:
                break
        taken.add(candidate)
        stems[stem] = candidate

    def _new_name(name: str, is_dir: bool) -> str:
        stem, ext = _split_ext(name, is_dir)
        return stems[stem] + ext if stem in stems else name

    renamed = {rel: is_dir for rel, is_dir in entries}
    mapping = []
    for rel, is_dir in entries:
        new_parts = []
        prefix = Path()
        for part in rel.parts:
            prefix = prefix / part
            new_parts.append(
                _new_name(part, renamed[prefix]) if prefix in renamed else part
            )
        mapping.append({"old": rel.as_posix(), "new": Path(*new_parts).as_posix()})
    # Deepest first, so a parent rename never invalidates a pending child.
    for rel, is_dir in sorted(entries, key=lambda e: len(e[0].parts), reverse=True):
        path = root / rel
        path.rename(path.with_name(_new_name(rel.name, is_dir)))

    ordered = sorted(stems, key=len, reverse=True)
    regex = re.compile(
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(re.escape(stem) for stem in ordered)
        + r")(?![A-Za-z0-9])"
    )
    for path, data in _iter_text_files(root):
        if data is None:
            continue
        text = _decode(data)
        new = _sub_stems(text, regex, stems)
        if new != text:
            _rewrite(path, new.encode("utf-8", errors="surrogateescape"))
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            link = Path(parent) / name
            if not link.is_symlink():
                continue
            target = os.readlink(link)
            new_target = _sub_stems(target, regex, stems)
            if new_target != target:
                link.unlink()
                link.symlink_to(new_target)
    return sorted(mapping, key=lambda row: row["old"])


def assert_no_identifier_paths(
    root: Path,
    exact: Iterable[str] = (),
    patterns: Iterable[str] = (),
    *,
    capsule: str,
) -> None:
    """Fail loud if any path component under ``root`` carries an identifier."""
    exact = list(exact)
    patterns = list(patterns)
    for rel, _is_dir in _walk_names(root):
        hits = path_identifier_hits(rel.name, exact, patterns)
        if hits:
            name, value = hits[0]
            raise IdentifierPathLeakError(
                f"capsule {capsule}: {name} {value!r} still present in path "
                f"component {rel.as_posix()!r}"
            )


__all__ = [
    "PATH_POLICY",
    "IdentifierPathLeakError",
    "assert_no_identifier_paths",
    "neutralize_identifier_paths",
    "path_identifier_hits",
]

# EOF
