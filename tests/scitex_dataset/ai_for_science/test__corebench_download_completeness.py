#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/ai_for_science/test__corebench_download_completeness.py

"""#104 — truncated / partial downloads are never reported as fetched.

Drives the PRODUCTION ``_http_download`` against a REAL local HTTP server
(``http.server`` on 127.0.0.1, in a thread) that serves synthetic capsule
tarballs complete, truncated with an honest ``Content-Length`` (connection
closed early), truncated without a ``Content-Length``, and truncated once
then complete (retry). No mocks, no network beyond loopback.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _corebench_download, corebench


def _targz_bytes(seed: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Incompressible-ish payload so a truncation lands mid-stream.
        data = hashlib.sha256(seed.encode()).hexdigest().encode() * 4096
        info = tarfile.TarInfo("code/data.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


GOOD = _targz_bytes("good")


class _Handler(BaseHTTPRequestHandler):
    hits: dict[str, int] = {}

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):  # noqa: N802 - http.server API
        name = self.path.rsplit("/", 1)[-1]
        n = self.hits[name] = self.hits.get(name, 0) + 1
        half = GOOD[: len(GOOD) // 2]
        if name.startswith("capsule-good") or (
            name.startswith("capsule-flaky") and n > 1
        ):
            body, declared = GOOD, len(GOOD)
        elif name.startswith("capsule-trunc") or name.startswith("capsule-flaky"):
            body, declared = half, len(GOOD)  # promises more than it sends
        elif name.startswith("capsule-nolen"):
            body, declared = half, None  # no length: only the archive check sees it
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        if declared is not None:
            self.send_header("Content-Length", str(declared))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True


@pytest.fixture
def server():
    _Handler.hits = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _download(server, raw, cid, **kw):
    return _corebench_download.download(
        raw_dir=raw, capsule_ids=[cid], base_url=server, **kw
    )


def _ledger(raw: Path) -> dict:
    return json.loads((raw / ".checksums.json").read_text())


# --- low-level helper -------------------------------------------------------


def test_http_download_rejects_short_body(server, tmp_path):
    # Arrange
    dest = tmp_path / "x.tar.gz"
    # Act
    raised = pytest.raises(_corebench_download.IncompleteDownloadError)
    # Assert
    with raised:
        _corebench_download._http_download(
            f"{server}/capsules/capsule-trunc.tar.gz", dest
        )


def test_http_download_leaves_no_partial_file(server, tmp_path):
    # Arrange
    dest = tmp_path / "x.tar.gz"
    # Act
    try:
        _corebench_download._http_download(
            f"{server}/capsules/capsule-trunc.tar.gz", dest
        )
    except _corebench_download.IncompleteDownloadError:
        pass
    # Assert
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_http_download_complete_body_is_byte_identical(server, tmp_path):
    # Arrange
    dest = tmp_path / "x.tar.gz"
    # Act
    _corebench_download._http_download(f"{server}/capsules/capsule-good.tar.gz", dest)
    # Assert
    assert dest.read_bytes() == GOOD


# --- download(): counted, retried, recorded ---------------------------------


def test_complete_capsule_counts_as_fetched(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    result = _download(server, raw, "capsule-good")
    # Assert
    assert (result["n_fetched"], result["n_failed"]) == (1, 0)


def test_truncated_capsule_is_not_fetched(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    result = _download(server, raw, "capsule-trunc")
    # Assert
    assert (result["n_fetched"], result["n_failed"]) == (0, 1)


def test_truncated_capsule_leaves_no_file(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    _download(server, raw, "capsule-trunc")
    # Assert
    assert list((raw / "capsules").iterdir()) == []


def test_truncated_capsule_is_not_recorded(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    _download(server, raw, "capsule-trunc")
    # Assert
    assert _ledger(raw) == {}


def test_truncated_capsule_is_retried(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    _download(server, raw, "capsule-trunc")
    # Assert
    assert _Handler.hits["capsule-trunc.tar.gz"] == _corebench_download._FETCH_ATTEMPTS


def test_missing_content_length_truncation_caught_by_archive_check(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    result = _download(server, raw, "capsule-nolen")
    # Assert
    assert (result["n_fetched"], result["n_failed"]) == (0, 1)


def test_transient_truncation_recovers_on_retry(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    result = _download(server, raw, "capsule-flaky")
    # Assert
    assert (result["n_fetched"], _Handler.hits["capsule-flaky.tar.gz"]) == (1, 2)


def test_published_checksum_mismatch_is_not_fetched(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    # Act
    result = _download(
        server, raw, "capsule-good", expected_sha256={"capsule-good": "0" * 64}
    )
    # Assert
    assert (result["n_fetched"], result["n_failed"]) == (0, 1)


def test_published_checksum_match_is_fetched(server, tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    digest = hashlib.sha256(GOOD).hexdigest()
    # Act
    result = _download(
        server, raw, "capsule-good", expected_sha256={"capsule-good": digest}
    )
    # Assert
    assert result["n_fetched"] == 1


def test_leftover_partial_from_older_run_is_refetched(server, tmp_path):
    # Arrange — an unrecorded half file, as an older release could leave.
    raw = tmp_path / "raw"
    (raw / "capsules").mkdir(parents=True)
    (raw / "capsules" / "capsule-good.tar.gz").write_bytes(GOOD[: len(GOOD) // 2])
    # Act
    result = _download(server, raw, "capsule-good")
    # Assert
    assert (result["n_have"], result["n_remismatch"], result["n_fetched"]) == (0, 1, 1)


def test_complete_unrecorded_file_is_kept_and_recorded(server, tmp_path):
    # Arrange — a hand-staged complete tarball is verified, not re-fetched.
    raw = tmp_path / "raw"
    (raw / "capsules").mkdir(parents=True)
    (raw / "capsules" / "capsule-good.tar.gz").write_bytes(GOOD)
    # Act
    result = _download(server, raw, "capsule-good")
    # Assert
    assert (result["n_have"], "capsules/capsule-good.tar.gz" in _ledger(raw)) == (
        1,
        True,
    )


def test_prepare_fails_loud_on_failed_download(tmp_path):
    # Arrange — the module seam (save/restore, no mock) delivers half a
    # tarball, as a dropped connection without Content-Length would.
    def _half(url, dest):
        Path(dest).write_bytes(GOOD[: len(GOOD) // 2])

    saved = _corebench_download._http_download
    _corebench_download._http_download = _half
    # Act
    raised = pytest.raises(corebench.IncompleteDownloadError)
    # Assert
    try:
        with raised:
            corebench.prepare(dataset_root=tmp_path, capsule_ids=["capsule-x"])
    finally:
        _corebench_download._http_download = saved


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
