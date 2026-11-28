#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_fetch_file.py

"""Checksum-verified, cached single-file download with a provenance record.

:func:`fetch_file` is the generic primitive behind
:func:`scitex_dataset.fetch_reference`: download one known file, verify its
sha256, keep it in ``dest_dir`` and say where it came from.

Contract (no fallback, no silent skip):

- ``dest_dir/filename`` already present with the expected sha256 → nothing is
  downloaded; the record has ``cached=True``.
- ``dest_dir/filename`` present with a DIFFERENT sha256 → :class:`FetchError`,
  unless ``replace=True`` (a user's file is never overwritten implicitly).
- Otherwise the body is streamed to a ``.part`` file inside ``dest_dir``; the
  byte count must match a declared ``Content-Length`` and the sha256 must
  match. Only then is it renamed into place (``os.replace``, atomic on one
  filesystem). On any failure the partial file is removed and
  :class:`FetchError` (or the underlying network error) is raised.

Stdlib only (``urllib``), like the CORE-Bench capsule downloader.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

_CHUNK = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_USER_AGENT = "scitex-dataset (fetch_file)"


class FetchError(RuntimeError):
    """A file could not be fetched with the expected content.

    Raised on a checksum mismatch (downloaded or pre-existing file) and on a
    truncated body. The partial download is always removed first.
    """


@dataclass(frozen=True)
class FetchRecord:
    """Provenance of one fetched file.

    Fields (all plain JSON types, see :meth:`to_dict`):

    - ``url`` — where the bytes came from (or would have, if cached).
    - ``sha256`` — verified lowercase hex digest of the file at ``path``.
    - ``size_bytes`` — file size in bytes.
    - ``path`` — absolute local path of the verified file.
    - ``retrieved_at`` — ISO-8601 UTC timestamp of this call (download or
      cache check).
    - ``cached`` — True when the file was already present and verified;
      nothing was downloaded.
    - ``license`` / ``citation`` — content license and preferred citation,
      when known (else None).
    """

    url: str
    sha256: str
    size_bytes: int
    path: str
    retrieved_at: str
    cached: bool
    license: Optional[str] = None
    citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a plain JSON-serialisable dict."""
        return asdict(self)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_from_url(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    if not name:
        raise ValueError(
            f"fetch_file: cannot derive a filename from {url!r}; pass filename="
        )
    return name


def fetch_file(
    url: str,
    sha256: str,
    dest_dir: Union[str, Path],
    filename: Optional[str] = None,
    *,
    timeout: float = 60.0,
    replace: bool = False,
    license: Optional[str] = None,
    citation: Optional[str] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> FetchRecord:
    """Download one file, verify its sha256, cache it, return its provenance.

    Parameters
    ----------
    url : str
        Source URL (``http``/``https``; ``file`` also works for local mirrors).
    sha256 : str
        Expected sha256 hex digest (64 hex chars, case-insensitive). Required:
        an unverified download is not a provenance record.
    dest_dir : str or Path
        Directory to keep the file in. Created if missing.
    filename : str, optional
        Local file name. Defaults to the last path segment of ``url``.
    timeout : float
        Socket timeout in seconds for the HTTP request.
    replace : bool
        If an existing file has the wrong sha256, replace it with a fresh
        download instead of raising. Default False.
    license, citation : str, optional
        Metadata copied into the returned record.
    progress : callable, optional
        Called as ``progress(bytes_done, bytes_total_or_None)`` after each
        chunk of a download (not for a cached file). An exception it raises
        aborts the download; the partial file is removed as on any failure.

    Returns
    -------
    FetchRecord
        ``url``, ``sha256``, ``size_bytes``, ``path``, ``retrieved_at`` (ISO
        UTC), ``cached``, ``license``, ``citation``.

    Raises
    ------
    ValueError
        Malformed ``sha256`` or underivable ``filename``.
    FetchError
        Checksum mismatch (download or existing file without ``replace``),
        or a body shorter than its declared ``Content-Length``.
    urllib.error.URLError
        Network / HTTP failure (propagated unchanged).
    """
    expected = sha256.strip().lower()
    if not _SHA256_RE.match(expected):
        raise ValueError(f"fetch_file: sha256 must be 64 hex chars, got {sha256!r}")
    name = filename or _filename_from_url(url)
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"fetch_file: filename must be a bare name, got {name!r}")

    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    meta = dict(url=url, license=license, citation=citation)

    if dest.exists():
        actual = _sha256_of(dest)
        if actual == expected:
            return FetchRecord(
                sha256=actual,
                size_bytes=dest.stat().st_size,
                path=str(dest),
                retrieved_at=_now_utc(),
                cached=True,
                **meta,
            )
        if not replace:
            raise FetchError(
                f"fetch_file: existing {dest} has sha256 {actual}, expected "
                f"{expected}. Refusing to overwrite; remove it or pass "
                f"replace=True."
            )

    part = dest_dir / f".{name}.part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        h = hashlib.sha256()
        written = 0
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            declared = resp.headers.get("Content-Length")
            with part.open("wb") as fh:
                total = int(declared) if declared is not None else None
                for chunk in iter(lambda: resp.read(_CHUNK), b""):
                    fh.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
        if declared is not None and written != int(declared):
            raise FetchError(
                f"fetch_file: truncated body from {url} ({written} of {declared} bytes)"
            )
        actual = h.hexdigest()
        if actual != expected:
            raise FetchError(
                f"fetch_file: sha256 mismatch for {url}: got {actual}, "
                f"expected {expected}. Partial download removed."
            )
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()

    return FetchRecord(
        sha256=expected,
        size_bytes=written,
        path=str(dest),
        retrieved_at=_now_utc(),
        cached=False,
        **meta,
    )


__all__ = ["FetchError", "FetchRecord", "fetch_file"]

# EOF
