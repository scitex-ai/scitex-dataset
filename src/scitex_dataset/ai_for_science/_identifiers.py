#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_identifiers.py

"""Hosting-identifier redaction for the agent-visible ``for_solver/`` view.

Answer masking (see :mod:`._standardize`) removes the *values* a solver must
reproduce. That is defeated when the shipped capsule still tells a solver
WHERE the published values live: a solver with network access can follow a
capsule page, a published environment image, the paper DOI, or a hosted
result figure and read the answer instead of reproducing it.

This module neutralises those pointers in every extracted capsule's text
files, in place, and provides a fail-loud backstop. Identifier classes
(ADR 0001, "Hosting identifiers"):

1. **Published environment images** — ``registry.codeocean.com/published/
   <uuid>:<tag>`` (optionally ``@sha256:...``) is the capsule's own
   archived image. It is replaced by :data:`LOCAL_ENV_IMAGE`, a local tag
   the solver can build from ``environment/Dockerfile``, so the
   ``docker run ...`` command stays usable.
2. **Environment digests** — the ``# hash:sha256:<hex>`` header line of an
   exported Dockerfile is removed (a comment; the Dockerfile still builds)
   and any other bare ``sha256:<hex>`` digest becomes ``[redacted-digest]``.
   ``image@sha256:...`` base-image pins are KEPT (needed to rebuild).
3. **Hosting/publication URLs** — any URL (with or without scheme) on a
   capsule-hosting, archive or publisher host in :data:`HOSTING_HOSTS`
   becomes ``[redacted-url]``. Generic base images
   (``registry.codeocean.com/codeocean/...``) are KEPT.
4. **Result-file links** — a remote image embedded in docs (Markdown
   ``![..](https://...)`` or HTML ``<img src="https://...">``) and any URL
   with a result-like path segment (:data:`RESULT_URL_SEGMENTS`) become
   ``[redacted-url]``: they render the published outputs, never inputs.
5. **DOIs** — bare ``10.NNNN/...`` DOIs become ``[redacted-doi]``.
6. **Exact capsule identifiers** — adapter-supplied strings (native capsule
   id, its UUID core, the paper DOI and title) are replaced
   case-insensitively by ``[redacted-identifier]``.

KEPT, because a solver needs them to reproduce: dependency lists
(``requirements.txt``, Dockerfile ``RUN`` lines, ``postInstall``), package
versions, base-image ``FROM`` lines, run commands/scripts, code-repository
and package-index URLs (``github.com``, ``pypi.org``), documentation links.

Bound: only text files (no NUL byte in the first 8 KiB) up to
:data:`TEXT_SIZE_LIMIT` are rewritten/scanned; symlinks are never followed.
Larger files are reported as ``unscanned``. Notebooks (``.ipynb``) are
additionally scanned/redacted per DECODED string (notebook and cell metadata,
sources, outputs, attachments), so a JSON escape cannot hide an identifier.
Identifiers in path NAMES are handled by :mod:`._neutral_paths`. Author names and the abstract
are not removed (licences/attribution); a solver that must not consult the
web needs a network-isolated sandbox — redaction is defence in depth.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

# Bump when any rule below changes: materialization identity includes it, so
# a cache built under an older (leakier) policy refuses reuse.
# v2: notebooks are also scanned/redacted STRUCTURALLY (every decoded string of
#     notebook- and cell-level metadata, sources, outputs and attachments, so a
#     JSON escape cannot hide an identifier), and path names are neutralised
#     (:mod:`._neutral_paths`, #111).
IDENTIFIER_POLICY = "hosting-identifiers-v2"

LOCAL_ENV_IMAGE = "capsule-environment:local"
REDACTED_URL = "[redacted-url]"
REDACTED_DOI = "[redacted-doi]"
REDACTED_DIGEST = "[redacted-digest]"
REDACTED_IDENTIFIER = "[redacted-identifier]"

TEXT_SIZE_LIMIT = 64 * 1_024 * 1_024
_SNIFF_BYTES = 8_192
_MIN_EXACT_LEN = 6

# (host suffix, path prefix). Empty prefix = whole host. A URL is a hosting
# identifier when its host equals/ends with ``.<suffix>`` and its path starts
# with the prefix (case-insensitive).
HOSTING_HOSTS: tuple[tuple[str, str], ...] = (
    # capsule / benchmark hosting
    ("codeocean.com", ""),
    ("corebench.cs.princeton.edu", ""),
    ("huggingface.co", "/datasets/futurehouse"),
    ("mybinder.org", ""),
    ("wholetale.org", ""),
    # DOI resolvers + research archives (host published results)
    ("doi.org", ""),
    ("zenodo.org", ""),
    ("figshare.com", ""),
    ("osf.io", ""),
    ("datadryad.org", ""),
    ("dataverse.harvard.edu", ""),
    ("researchgate.net", ""),
    ("semanticscholar.org", ""),
    ("paperswithcode.com", ""),
    # preprint servers / publishers (the paper reporting the answers)
    ("arxiv.org", ""),
    ("biorxiv.org", ""),
    ("medrxiv.org", ""),
    ("ieeexplore.ieee.org", ""),
    ("dl.acm.org", ""),
    ("springer.com", ""),
    ("sciencedirect.com", ""),
    ("nature.com", ""),
    ("science.org", ""),
    ("cell.com", ""),
    ("pnas.org", ""),
    ("plos.org", ""),
    ("wiley.com", ""),
    ("tandfonline.com", ""),
    ("mdpi.com", ""),
    ("frontiersin.org", ""),
    ("openreview.net", ""),
    ("aclanthology.org", ""),
    ("europepmc.org", ""),
    ("pubmed.ncbi.nlm.nih.gov", ""),
    ("ncbi.nlm.nih.gov", "/pmc"),
    ("ncbi.nlm.nih.gov", "/pubmed"),
)

# Generic, capsule-independent base images on a hosting host — needed to
# rebuild the environment, so never redacted.
KEPT_HOSTING_PATHS: tuple[tuple[str, str], ...] = (
    ("registry.codeocean.com", "/codeocean/"),
)

# A URL whose path has one of these segments points at published outputs.
RESULT_URL_SEGMENTS = frozenset(
    {"results", "result", "output", "outputs", "figures", "figure", "figs", "plots"}
)

_URL_CHARS = r"[^\s\"'<>`)\]]"
_PUBLISHED_IMAGE_RE = re.compile(
    r"\bregistry\.codeocean\.com/published/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"(?::[\w.-]+)?(?:@sha256:[0-9a-f]{64})?",
    re.IGNORECASE,
)
_DIGEST_HEADER_RE = re.compile(
    r"^[ \t]*#[ \t]*hash:[ \t]*sha256:[0-9a-f]{64}[ \t]*(?:\r?\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_BARE_DIGEST_RE = re.compile(r"(?<!@)\bsha256:[0-9a-f]{64}\b", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\(\s*)((?:https?|ftp)://" + _URL_CHARS + r"+)")
_HTML_IMG_RE = re.compile(
    r"(<img\b[^>]*?\bsrc\s*=\s*[\"']?)((?:https?|ftp)://[^\s\"'<>]+)", re.IGNORECASE
)
_SCHEME_URL_RE = re.compile(r"\b(?:https?|ftp)://" + _URL_CHARS + r"+", re.IGNORECASE)
_HOST_ALTERNATION = "|".join(
    sorted(
        {re.escape(h) for h, _ in HOSTING_HOSTS + KEPT_HOSTING_PATHS},
        key=len,
        reverse=True,
    )
)
_BARE_HOST_RE = re.compile(
    r"(?<![\w.@/-])(?:[a-z0-9-]+\.)*(?:"
    + _HOST_ALTERNATION
    + r")\b(?:/"
    + _URL_CHARS
    + r"*)?",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>`()\[\]{},;]+")
_TRAILING_PUNCT = ".,;:!?"
_UUID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)


class HostingIdentifierLeakError(RuntimeError):
    """A hosting identifier survived redaction in a scanned for_solver file."""


def _split_trailing(token: str) -> tuple[str, str]:
    core = token.rstrip(_TRAILING_PUNCT)
    return core, token[len(core) :]


def _host_path(url: str) -> tuple[str, str]:
    parts = urlsplit(url if "://" in url else "//" + url)
    return (parts.hostname or "").lower(), parts.path or "/"


def _host_matches(host: str, path: str, table) -> bool:
    path = path.lower()
    return any(
        (host == suffix or host.endswith("." + suffix)) and path.startswith(prefix)
        for suffix, prefix in table
    )


def is_hosting_url(url: str) -> bool:
    """True when ``url`` points at a hosting/publication/result location."""
    host, path = _host_path(url)
    if not host or _host_matches(host, path, KEPT_HOSTING_PATHS):
        return False
    if _host_matches(host, path, HOSTING_HOSTS):
        return True
    if "://" not in url:  # scheme-less: only listed hosts qualify
        return False
    return any(seg.lower() in RESULT_URL_SEGMENTS for seg in path.split("/"))


def native_identifiers(native_id: str | None) -> list[str]:
    """Exact identifiers derived from a native capsule id (id + UUID core)."""
    if not native_id:
        return []
    found = [native_id]
    found.extend(m.group(0) for m in _UUID_RE.finditer(native_id))
    return found


def publication_identifiers(doi: object = None, title: object = None) -> list[str]:
    """Exact identifiers of the publication whose results are the answers.

    ``doi`` may be a bare DOI or a resolver URL; both the versioned and the
    unversioned (``.vN`` stripped) DOI are returned. ``title`` is returned
    as-is (matched case-insensitively, whitespace-tolerant).
    """
    found: list[str] = []
    if isinstance(doi, str) and doi.strip():
        match = _DOI_RE.search(doi)
        bare = _split_trailing(match.group(0))[0] if match else doi.strip()
        found.append(bare)
        unversioned = re.sub(r"\.v\d+$", "", bare, flags=re.IGNORECASE)
        if unversioned != bare:
            found.append(unversioned)
    if isinstance(title, str) and title.strip():
        found.append(title.strip())
    return found


def _exact_patterns(exact: Iterable[str]) -> list[re.Pattern[str]]:
    patterns = []
    for value in sorted(
        {v.strip() for v in exact if isinstance(v, str)}, key=len, reverse=True
    ):
        if len(value) < _MIN_EXACT_LEN:
            continue
        body = r"\s+".join(re.escape(word) for word in value.split())
        lead = r"(?<![\w])" if value[0].isalnum() else ""
        tail = r"(?![\w])" if value[-1].isalnum() else ""
        patterns.append(re.compile(lead + body + tail, re.IGNORECASE))
    return patterns


def redact_text(text: str, exact: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Return ``(redacted text, identifier classes that changed)``."""
    classes: list[str] = []

    def _apply(name, pattern, repl):
        nonlocal text
        new = pattern.sub(repl, text)
        if new != text:
            classes.append(name)
            text = new

    def _url_repl(match):
        core, tail = _split_trailing(match.group(0))
        return (REDACTED_URL + tail) if is_hosting_url(core) else match.group(0)

    _apply("published_image", _PUBLISHED_IMAGE_RE, LOCAL_ENV_IMAGE)
    _apply("environment_digest", _DIGEST_HEADER_RE, "")
    _apply("environment_digest", _BARE_DIGEST_RE, REDACTED_DIGEST)
    _apply("result_link", _MD_IMAGE_RE, lambda m: m.group(1) + REDACTED_URL)
    _apply("result_link", _HTML_IMG_RE, lambda m: m.group(1) + REDACTED_URL)
    _apply("hosting_url", _SCHEME_URL_RE, _url_repl)
    _apply("hosting_url", _BARE_HOST_RE, _url_repl)
    _apply("doi", _DOI_RE, lambda m: REDACTED_DOI + _split_trailing(m.group(0))[1])
    for pattern in _exact_patterns(exact):
        _apply("exact_identifier", pattern, REDACTED_IDENTIFIER)
    return text, sorted(set(classes))


def find_identifiers(text: str, exact: Iterable[str] = ()) -> list[tuple[str, str]]:
    """List every ``(class, matched text)`` hosting identifier in ``text``."""
    hits: list[tuple[str, str]] = []
    simple = (
        ("published_image", _PUBLISHED_IMAGE_RE),
        ("environment_digest", _DIGEST_HEADER_RE),
        ("environment_digest", _BARE_DIGEST_RE),
        ("doi", _DOI_RE),
    )
    for name, pattern in simple:
        hits.extend((name, m.group(0).strip()) for m in pattern.finditer(text))
    for name, pattern in (("result_link", _MD_IMAGE_RE), ("result_link", _HTML_IMG_RE)):
        hits.extend((name, m.group(2)) for m in pattern.finditer(text))
    for pattern in (_SCHEME_URL_RE, _BARE_HOST_RE):
        for m in pattern.finditer(text):
            core = _split_trailing(m.group(0))[0]
            if is_hosting_url(core):
                hits.append(("hosting_url", core))
    for pattern in _exact_patterns(exact):
        hits.extend(("exact_identifier", m.group(0)) for m in pattern.finditer(text))
    return hits


def _iter_text_files(root: Path):
    """Yield ``(path, raw bytes)`` for regular text files; skipped → ``None``."""
    for parent, _dirs, files in os.walk(root, followlinks=False):
        for name in sorted(files):
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                size = path.stat().st_size
                with path.open("rb") as fh:
                    head = fh.read(_SNIFF_BYTES)
                if b"\0" in head:
                    continue  # binary
                if size > TEXT_SIZE_LIMIT:
                    yield path, None
                    continue
                yield path, path.read_bytes()
            except OSError:
                yield path, None


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape")


def _rewrite(path: Path, data: bytes) -> None:
    """Rewrite in place, keeping the mode; a read-only member (common in
    published archives) is made owner-writable only for the write."""
    mode = stat.S_IMODE(path.stat().st_mode)
    if not mode & stat.S_IWUSR:
        path.chmod(mode | stat.S_IWUSR)
    try:
        path.write_bytes(data)
    finally:
        path.chmod(mode)


_NOTEBOOK_EXT = ".ipynb"


def _load_notebook(text: str):
    try:
        nb = json.loads(text)
    except ValueError:
        return None
    return nb if isinstance(nb, dict) else None


def _json_strings(node):
    """Yield every decoded string (keys AND values) of a parsed JSON node."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _json_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _json_strings(value)


def _redact_json(node, exact, classes: set):
    if isinstance(node, str):
        new, found = redact_text(node, exact)
        classes.update(found)
        return new
    if isinstance(node, dict):
        return {
            _redact_json(k, exact, classes): _redact_json(v, exact, classes)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_redact_json(v, exact, classes) for v in node]
    return node


def _notebook_identifiers(text: str, exact) -> list[tuple[str, str]]:
    """Identifiers in a notebook's DECODED strings: notebook-level metadata,
    cell metadata, sources, outputs and attachments (escapes cannot hide)."""
    nb = _load_notebook(text)
    if nb is None:
        return []
    hits: list[tuple[str, str]] = []
    for value in _json_strings(nb):
        hits.extend(find_identifiers(value, exact))
    return hits


def _redact_notebook(text: str, exact) -> tuple[str, list[str]]:
    """Structurally redact a notebook whose decoded strings still carry ids."""
    if not _notebook_identifiers(text, exact):
        return text, []
    nb = _load_notebook(text)
    classes: set = set()
    nb = _redact_json(nb, exact, classes)
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n", sorted(classes)


def mask_hosting_identifiers(input_dir: Path, exact: Iterable[str] = ()) -> dict:
    """Redact hosting identifiers in every text file under ``input_dir``.

    Returns ``{"redacted": [paths], "classes": {class: n_files},
    "unscanned": [paths]}``. File modes are preserved (in-place rewrite).
    """
    exact = list(exact)
    redacted: list[str] = []
    unscanned: list[str] = []
    counts: dict[str, int] = {}
    for path, data in _iter_text_files(input_dir):
        if data is None:
            unscanned.append(str(path))
            continue
        new, classes = redact_text(_decode(data), exact)
        if path.suffix.lower() == _NOTEBOOK_EXT:
            new, nb_classes = _redact_notebook(new, exact)
            classes = sorted(set(classes) | set(nb_classes))
        if not classes:
            continue
        _rewrite(path, new.encode("utf-8", errors="surrogateescape"))
        redacted.append(str(path))
        for name in classes:
            counts[name] = counts.get(name, 0) + 1
    return {"redacted": redacted, "classes": counts, "unscanned": unscanned}


def assert_no_hosting_identifiers(
    input_dir: Path, exact: Iterable[str] = (), *, capsule: str
) -> None:
    """Fail loud if any scanned text file still carries a hosting identifier."""
    exact = list(exact)
    for path, data in _iter_text_files(input_dir):
        if data is None:
            continue
        text = _decode(data)
        hits = find_identifiers(text, exact)
        if not hits and path.suffix.lower() == _NOTEBOOK_EXT:
            hits = _notebook_identifiers(text, exact)
        if hits:
            name, value = hits[0]
            raise HostingIdentifierLeakError(
                f"capsule {capsule}: {name} {value!r} still present in {path}"
            )


__all__ = [
    "IDENTIFIER_POLICY",
    "LOCAL_ENV_IMAGE",
    "HOSTING_HOSTS",
    "KEPT_HOSTING_PATHS",
    "RESULT_URL_SEGMENTS",
    "HostingIdentifierLeakError",
    "assert_no_hosting_identifiers",
    "find_identifiers",
    "is_hosting_url",
    "mask_hosting_identifiers",
    "native_identifiers",
    "publication_identifiers",
    "redact_text",
]

# EOF
