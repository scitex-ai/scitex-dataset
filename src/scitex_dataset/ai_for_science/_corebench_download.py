#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_corebench_download.py

"""CORE-Bench network + IO side: oracle bootstrap and capsule download.

Split out of :mod:`.corebench` (the 512-line-per-file guard). This module
owns everything that touches the network or the operator-private
``raw_dir`` on disk:

- ``bootstrap_oracle(...)`` — fetch the upstream answer manifests
  (``core_train.json`` plaintext + ``core_test.json.gpg`` ciphertext) and
  gpg-decrypt the test split into ``raw_dir`` with the README-published
  passphrase. The staging layout is ASYMMETRIC and matches the readers in
  :mod:`.corebench`:

    * ``raw/dataset/core_train.json``    (plaintext, pristine)
    * ``raw/dataset/core_test.json.gpg`` (ciphertext, kept pristine)
    * ``raw/core_test.json``             (decrypted, at raw/ ROOT)

- ``download(...)`` — pull capsule tarballs into ``raw_dir/capsules/`` with
  an sha256 skip ledger; bootstraps the oracle first when the caller did
  not pass an explicit ``capsule_ids`` list.
- ``_decrypt_gpg(...)`` — module-level, seam-swappable gpg wrapper.

The decrypted answers live ONLY in operator-private ``raw_dir`` and are
never copied into ``for_solver/`` (leak guard lives in the standardizer).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zlib
from pathlib import Path
from typing import Iterable

from ._manifest import sha256_file

# Princeton CDN hosting the capsule tarballs.
SOURCE_URL = "https://corebench.cs.princeton.edu"

# Upstream oracle manifests (siegelz/core-bench, pinned to the ``main`` ref).
_ORACLE_UPSTREAM_REF = "main"
_ORACLE_TRAIN_URL = (
    "https://raw.githubusercontent.com/siegelz/core-bench/"
    f"{_ORACLE_UPSTREAM_REF}/benchmark/dataset/core_train.json"
)
_ORACLE_TEST_GPG_URL = (
    "https://raw.githubusercontent.com/siegelz/core-bench/"
    f"{_ORACLE_UPSTREAM_REF}/benchmark/dataset/core_test.json.gpg"
)
# Symmetric passphrase for the encrypted test split — published in the
# upstream README (``reproducibility``). Not a secret; it gates casual
# answer-scraping, not the operator.
_ORACLE_TEST_PASSPHRASE = "reproducibility"

# Staging relpaths under raw_dir (ASYMMETRIC — see module docstring). The
# decrypted test manifest lands at the raw/ ROOT because the standardizer
# reads it from ``_ORACLE_TEST_RELPATH = ("core_test.json",)``.
_ORACLE_TRAIN_RELPATH = ("dataset", "core_train.json")
_ORACLE_TEST_GPG_RELPATH = ("dataset", "core_test.json.gpg")
_ORACLE_TEST_RELPATH = ("core_test.json",)

# Bulk capsule tarballs under raw_dir.
_CAPSULES_SUBDIR = "capsules"

# sha256 ledger recording the integrity of fetched capsule tarballs. Lives
# in operator-private raw_dir so re-runs can skip already-verified files.
# NOTE: the oracle manifests are NOT recorded here — this ledger's relpaths
# are treated as capsule tarballs by verify_integrity.
_CHECKSUMS_FILENAME = ".checksums.json"


def _capsule_url(base_url: str, capsule_id: str) -> str:
    return f"{base_url}/capsules/{capsule_id}.tar.gz"


class IncompleteDownloadError(RuntimeError):
    """A fetched file is truncated, corrupt or fails its published checksum.

    Raised instead of recording the file as fetched; the partial bytes are
    removed so a re-run never mistakes them for a complete download.
    """


# Attempts per capsule before it is reported as failed (transient truncation).
_FETCH_ATTEMPTS = 3


def _http_download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` via stdlib urllib, verified and atomic.

    Streams in 1 MiB chunks to ``<dest>.part``; when the server declares a
    ``Content-Length`` the byte count written must equal it (urllib does NOT
    raise on a connection closed early — it just returns EOF), else
    :class:`IncompleteDownloadError`. Only a complete body is renamed onto
    ``dest`` (``os.replace``), so ``dest`` never holds a partial file and any
    leftover ``.part`` is removed on failure. Caller is responsible for
    ``dest.parent.mkdir``.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url) as resp, part.open("wb") as fh:
            declared = resp.headers.get("Content-Length")
            written = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
        if declared is not None and written != int(declared):
            raise IncompleteDownloadError(
                f"{url}: truncated body ({written} of {declared} bytes)"
            )
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()


def _verify_targz(path: Path) -> None:
    """Read a ``.tar.gz`` end to end: tar headers, member bodies, gzip CRC.

    A truncated or corrupt archive raises :class:`IncompleteDownloadError`.
    Streaming mode (``r|gz``) reads every byte once, so the gzip trailer's
    CRC-32 + length check runs as well; nothing is extracted.
    """
    try:
        with tarfile.open(path, mode="r|gz") as tf:
            for member in tf:
                body = tf.extractfile(member) if member.isfile() else None
                if body is not None:
                    while body.read(1024 * 1024):
                        pass
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        raise IncompleteDownloadError(
            f"{path.name}: archive integrity check failed ({exc})"
        ) from exc


def _fetch_capsule(url: str, out: Path, expected_sha256: str | None) -> str:
    """Fetch + verify one capsule tarball, retrying; return its sha256.

    Each attempt: :func:`_http_download` (Content-Length checked, atomic),
    :func:`_verify_targz`, then the published sha256 when one is supplied.
    A failed attempt removes ``out``; after :data:`_FETCH_ATTEMPTS` the last
    error propagates (the caller counts it as failed, never as fetched).
    """
    last: Exception | None = None
    for _ in range(_FETCH_ATTEMPTS):
        try:
            _http_download(url, out)
            _verify_targz(out)
            digest = sha256_file(out)
            if expected_sha256 is not None and digest != expected_sha256.lower():
                raise IncompleteDownloadError(
                    f"{out.name}: sha256 {digest} != published {expected_sha256}"
                )
            return digest
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            if out.exists():
                out.unlink()
    assert last is not None
    raise last


def _nonempty(path: Path) -> bool:
    """True if ``path`` exists as a non-empty file."""
    return path.is_file() and path.stat().st_size > 0


def _load_checksums(raw_dir: Path) -> dict:
    """Read ``raw_dir/.checksums.json`` ({relpath: sha256}); {} if absent."""
    ledger = raw_dir / _CHECKSUMS_FILENAME
    if not ledger.is_file():
        return {}
    try:
        return json.loads(ledger.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover — corrupt ledger
        return {}


def _save_checksums(raw_dir: Path, checksums: dict) -> None:
    """Write the sha256 ledger back to ``raw_dir/.checksums.json``."""
    ledger = raw_dir / _CHECKSUMS_FILENAME
    ledger.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Oracle bootstrap — fetch + gpg-decrypt the answer manifests.
# ---------------------------------------------------------------------------


def _decrypt_gpg(src_gpg: Path, dest: Path, passphrase: str) -> None:
    """Symmetrically decrypt ``src_gpg`` to ``dest`` with ``passphrase``.

    The passphrase is written to gpg over **stdin (fd 0)** — never on argv
    (which would leak via ``ps``). ``--pinentry-mode loopback`` is REQUIRED
    on GnuPG >= 2.1 under ``--batch`` to accept the passphrase on a pipe.

    On a missing gpg binary → RuntimeError telling the operator to install
    it or hand-stage the plaintext. On a non-zero exit → the partial
    ``--output`` (gpg leaves a 0-byte dest) is unlinked and a RuntimeError
    carrying the exit code + stderr is raised (fail-loud).
    """
    gpg = shutil.which("gpg") or shutil.which("gpg2")
    if gpg is None:
        raise RuntimeError(
            "corebench.bootstrap_oracle: gpg not found on PATH. Install "
            "gnupg (e.g. `apt install gnupg` / `brew install gnupg`) or "
            f"hand-stage the decrypted manifest at {dest}."
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            gpg,
            "--batch",
            "--yes",
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--output",
            str(dest),
            "--decrypt",
            str(src_gpg),
        ],
        input=passphrase.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"corebench: gpg decrypt failed (exit {proc.returncode}) for "
            f"{src_gpg}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )


def oracle_staged(raw_dir: Path) -> bool:
    """Both oracle manifests standardize reads (train + decrypted test) are non-empty."""
    raw_dir = Path(raw_dir)
    return all(
        _nonempty(raw_dir.joinpath(*rel))
        for rel in (_ORACLE_TRAIN_RELPATH, _ORACLE_TEST_RELPATH)
    )


def bootstrap_oracle(raw_dir: Path, *, force: bool = False) -> dict:
    """Fetch + stage the CORE-Bench oracle manifests into ``raw_dir``.

    Idempotent: a stage whose target is already non-empty is skipped unless
    ``force=True``. If only the decrypted ``raw/core_test.json`` is missing
    but the ``.gpg`` ciphertext is present, it is re-decrypted from the kept
    ciphertext (no re-fetch). The manifests are NOT recorded in the capsule
    checksum ledger.

    Returns a dict describing the staged paths and which stages ran.
    """
    raw_dir = Path(raw_dir)
    train = raw_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test_gpg = raw_dir.joinpath(*_ORACLE_TEST_GPG_RELPATH)
    test_plain = raw_dir.joinpath(*_ORACLE_TEST_RELPATH)

    staged = {"train": False, "test_gpg": False, "test_decrypted": False}

    # Train split — plaintext, pristine under raw/dataset/.
    if force or not _nonempty(train):
        train.parent.mkdir(parents=True, exist_ok=True)
        _http_download(_ORACLE_TRAIN_URL, train)
        staged["train"] = True

    # Test split — ciphertext, kept pristine under raw/dataset/.
    if force or not _nonempty(test_gpg):
        test_gpg.parent.mkdir(parents=True, exist_ok=True)
        _http_download(_ORACLE_TEST_GPG_URL, test_gpg)
        staged["test_gpg"] = True

    # Decrypt to raw/ ROOT (asymmetric). Re-decrypt when only the plaintext
    # is missing but the ciphertext is on disk.
    if force or not _nonempty(test_plain):
        _decrypt_gpg(test_gpg, test_plain, _ORACLE_TEST_PASSPHRASE)
        staged["test_decrypted"] = True

    return {
        "raw_dir": str(raw_dir),
        "train": str(train),
        "test_gpg": str(test_gpg),
        "test": str(test_plain),
        "staged": staged,
    }


# ---------------------------------------------------------------------------
# Download — network. Pulls capsule tarballs into ``raw_dir/capsules/``.
# ---------------------------------------------------------------------------


def _oracle_capsule_ids(raw_dir: Path) -> list[str]:
    """Read the canonical de-duplicated capsule-id list from the manifests."""
    train = raw_dir.joinpath(*_ORACLE_TRAIN_RELPATH)
    test = raw_dir.joinpath(*_ORACLE_TEST_RELPATH)
    seen: set[str] = set()
    ids: list[str] = []
    for src in (train, test):
        for entry in json.loads(src.read_text(encoding="utf-8")):
            cid = entry["capsule_id"]
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    ids.sort()
    return ids


def download(
    *,
    raw_dir: Path,
    capsule_ids: Iterable[str] | None = None,
    base_url: str = SOURCE_URL,
    verify_integrity: bool = False,
    force: bool = False,
    expected_sha256: dict[str, str] | None = None,
    **_,
) -> dict:
    """Pull capsule tarballs into ``raw_dir/capsules/``; skip what's complete.

    If ``capsule_ids`` is None the oracle manifests are bootstrapped first
    (fetched + gpg-decrypted via :func:`bootstrap_oracle`, ``force`` passed
    through), then the canonical 90-id list is read from the now-staged
    ``raw_dir/dataset/core_train.json`` + ``raw_dir/core_test.json``. When
    ``capsule_ids`` IS given explicitly the oracle URLs are never touched
    (the capsule-only path stays oracle-free).

    Completeness (#104) — a capsule counts as fetched ONLY after it passed
    every check: the body length equals the declared ``Content-Length``
    (written to ``.part`` and renamed atomically), the ``.tar.gz`` reads end
    to end (tar headers, member bodies, gzip CRC), and, when the caller
    supplies a published digest in ``expected_sha256`` ({capsule_id: hex}),
    the sha256 matches. A failed check is retried (:data:`_FETCH_ATTEMPTS`
    total), then the capsule is counted in ``n_failed`` with no file left
    behind. The ``raw_dir/.checksums.json`` ledger records ONLY verified
    capsules and is saved after each one.

    Skip policy (idempotent re-runs never re-download a complete capsule):

    - **default** — a capsule present AND recorded in the ledger is skipped
      with NO hashing (``n_have``). A present but UNRECORDED file (staged by
      hand, or a partial left by an older release) is archive-checked first:
      complete → recorded + ``n_have``; truncated/corrupt → re-fetched
      (``n_remismatch``).
    - ``verify_integrity=True`` — an existing capsule is sha256-checked
      against the ledger; a match is a verified skip (``n_skipped_verified``),
      a miss/drift is re-fetched (``n_remismatch``).
    - ``force=True`` — re-fetch everything regardless (and re-bootstrap the
      oracle manifests).

    HEAVY: ~13 GB across 90 tarballs. SLURM-only on shared compute;
    never call from a login node or CI.
    """
    raw_dir = Path(raw_dir)
    capsules_dir = raw_dir / _CAPSULES_SUBDIR
    capsules_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_sha256 or {}

    oracle = None
    if capsule_ids is None:
        oracle = bootstrap_oracle(raw_dir, force=force)
        capsule_ids = _oracle_capsule_ids(raw_dir)

    checksums = _load_checksums(raw_dir)
    n_have = n_skipped_verified = n_get = n_fail = n_remismatch = 0
    fails: list[str] = []
    for cid in capsule_ids:
        out = capsules_dir / f"{cid}.tar.gz"
        rel = out.relative_to(raw_dir).as_posix()
        if out.exists() and out.stat().st_size > 0 and not force:
            recorded = checksums.get(rel)
            if not verify_integrity and recorded is not None:
                # Default: verified on fetch, recorded → skip, no hashing.
                n_have += 1
                continue
            if verify_integrity and recorded is not None:
                if recorded == sha256_file(out):
                    n_skipped_verified += 1
                    continue
            elif not verify_integrity and _is_complete_archive(out):
                checksums[rel] = sha256_file(out)
                _save_checksums(raw_dir, checksums)
                n_have += 1
                continue
            # Unrecorded + incomplete, or drifted under verify: re-fetch below.
            n_remismatch += 1
        checksums.pop(rel, None)
        url = _capsule_url(base_url, cid)
        try:
            checksums[rel] = _fetch_capsule(url, out, expected.get(cid))
            n_get += 1
        except Exception as exc:
            if out.exists():
                out.unlink()
            n_fail += 1
            fails.append(f"{url}: {exc!r}")
        _save_checksums(raw_dir, checksums)

    _save_checksums(raw_dir, checksums)
    result = {
        "raw_dir": str(raw_dir),
        "capsules_dir": str(capsules_dir),
        "n_have": n_have,
        "n_skipped_verified": n_skipped_verified,
        "n_fetched": n_get,
        "n_remismatch": n_remismatch,
        "n_failed": n_fail,
        "failures": fails,
    }
    if oracle is not None:
        result["oracle"] = oracle
    return result


def _is_complete_archive(path: Path) -> bool:
    try:
        _verify_targz(path)
    except IncompleteDownloadError:
        return False
    return True


__all__ = [
    "SOURCE_URL",
    "IncompleteDownloadError",
    "bootstrap_oracle",
    "download",
]

# EOF
