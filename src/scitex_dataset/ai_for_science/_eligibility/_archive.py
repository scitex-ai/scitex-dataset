#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_archive.py

"""Read one unit's archive: member names plus its small code/env text files."""

from __future__ import annotations

import posixpath
import re
import tarfile
import zipfile
from pathlib import Path

# Read at most this many bytes of one text member (notebooks carry outputs).
_MAX_TEXT_BYTES = 64 * 1024 * 1024

_ENV_NAME = re.compile(
    r"(^|/)(dockerfile[^/]*|[^/]*requirements[^/]*\.txt|environment\.ya?ml|postinstall)$",
    re.I,
)
_ENTRY_SCRIPT_NAMES = ("run", "run.sh")


def _norm(path: str) -> str:
    """Archive member path without ``./`` or leading ``/`` (dotfiles kept)."""
    path = posixpath.normpath(path.replace("\\", "/"))
    while path.startswith(("./", "/")):
        path = path[2:] if path.startswith("./") else path[1:]
    return path


def _strip_common_root(names: list[str]) -> str:
    """Return the single top directory every member shares (``""`` if none)."""
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and all("/" in n for n in names):
        return tops.pop() + "/"
    return ""


def _wanted(name: str) -> bool:
    low = name.lower()
    base = posixpath.basename(low)
    return bool(
        _ENV_NAME.search(low)
        or base in _ENTRY_SCRIPT_NAMES
        or low.endswith((".py", ".ipynb", ".r", ".rmd", ".sh"))
    )


def read_archive_tree(
    archive: Path, *, member_prefix: str | None = None
) -> tuple[list[str], dict[str, bytes]]:
    """Read member names and the small code/env text files of an archive.

    ``member_prefix`` keeps only members under that directory (one unit
    inside a multi-unit archive) and strips it. A single top directory
    shared by every member is stripped too, so paths read ``code/run``
    whatever the archive's root folder is called.
    """
    archive = Path(archive)
    entries: list[tuple[str, int, object]] = []
    texts: dict[str, bytes] = {}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            names = [_norm(i.filename) for i in infos]
            if member_prefix is not None:
                pre = member_prefix.strip("/") + "/"
                keep = [(n, i) for n, i in zip(names, infos) if n.startswith(pre)]
                names = [n[len(pre) :] for n, _ in keep]
                infos = [i for _, i in keep]
            root = _strip_common_root(names)
            for name, info in zip(names, infos):
                rel = name[len(root) :]
                entries.append((rel, info.file_size, None))
                if _wanted(rel) and info.file_size <= _MAX_TEXT_BYTES:
                    texts[rel] = zf.read(info)
    else:
        raw: list[tuple[str, bytes | None]] = []
        with tarfile.open(archive, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = _norm(member.name)
                body = None
                if _wanted(name) and member.size <= _MAX_TEXT_BYTES:
                    fh = tf.extractfile(member)
                    body = fh.read() if fh is not None else None
                raw.append((name, body))
        names = [n for n, _ in raw]
        if member_prefix is not None:
            pre = member_prefix.strip("/") + "/"
            raw = [(n[len(pre) :], b) for n, b in raw if n.startswith(pre)]
            names = [n for n, _ in raw]
        root = _strip_common_root(names)
        for name, body in raw:
            rel = name[len(root) :]
            entries.append((rel, 0, None))
            if body is not None:
                texts[rel] = body
    return sorted(e[0] for e in entries), texts


__all__ = ["read_archive_tree"]

# EOF
