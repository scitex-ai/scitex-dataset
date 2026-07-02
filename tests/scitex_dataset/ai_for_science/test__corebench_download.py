#!/usr/bin/env python3
"""Tests for ai_for_science._corebench_download — oracle bootstrap + fetch.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The two
network/IO seams — ``_http_download`` (URL-keyed) and ``_decrypt_gpg`` —
are swapped via save/restore contextmanagers that assign the module
attribute and restore it. A dedicated real-gpg round-trip class exercises
the PRODUCTION ``_decrypt_gpg`` (skipped when gpg is absent).
"""

import json
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _base, _corebench_download, corebench

# Sentinel bytes the http stub writes for the ciphertext; the fake decrypt
# asserts it received exactly these (proving the .gpg was staged pristine).
_SENTINEL = b"CORE-BENCH-CIPHERTEXT-SENTINEL"


# ---------------------------------------------------------------------------
# Seams — save/restore swaps (no unittest.mock)
# ---------------------------------------------------------------------------


@contextmanager
def _swap(attr, replacement):
    saved = getattr(_corebench_download, attr)
    setattr(_corebench_download, attr, replacement)
    try:
        yield
    finally:
        setattr(_corebench_download, attr, saved)


@contextmanager
def _swapped_seams(http, dec):
    with _swap("_http_download", http), _swap("_decrypt_gpg", dec):
        yield


def _train_record():
    return {
        "capsule_id": "capsule-1111111",
        "language": "Python",
        "field": "biology",
        "task_prompt": "Read the README.",
        "results": [{"What is the AUC?": 0.81}],
    }


def _test_record():
    return {
        "capsule_id": "capsule-2222222",
        "language": "R",
        "field": "ecology",
        "task_prompt": "Run the R script.",
        "results": [{"What is the mean?": 4.2}],
    }


def _write_capsule_targz(path: Path, cid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / f"_scratch_{cid}"
    (scratch / "code").mkdir(parents=True, exist_ok=True)
    (scratch / "code" / "main.py").write_text(f"# {cid}\nprint('hi')\n")
    (scratch / "ReadMe").write_text(f"{cid} readme\n")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(scratch / "code" / "main.py", arcname="code/main.py")
        tf.add(scratch / "ReadMe", arcname="ReadMe")


class _UrlKeyedHttp:
    """URL-keyed HTTP stub writing real bytes per upstream endpoint."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append((url, str(dest)))
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("core_train.json"):
            dest.write_text(json.dumps([_train_record()]))
        elif url.endswith("core_test.json.gpg"):
            dest.write_bytes(_SENTINEL)
        elif url.endswith(".tar.gz"):
            cid = url.rsplit("/", 1)[-1][: -len(".tar.gz")]
            _write_capsule_targz(dest, cid)
        else:  # pragma: no cover — guards against an unexpected endpoint
            raise AssertionError(f"unexpected url: {url}")

    def n_gpg_fetches(self):
        return sum(1 for u, _ in self.calls if u.endswith(".gpg"))


class _FakeDecrypt:
    """Fake gpg seam: asserts the staged ciphertext + published passphrase."""

    def __init__(self):
        self.calls = 0

    def __call__(self, src_gpg, dest, passphrase):
        self.calls += 1
        assert Path(src_gpg).read_bytes() == _SENTINEL
        assert passphrase == "reproducibility"
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(json.dumps([_test_record()]))


# ---------------------------------------------------------------------------
# bootstrap_oracle — staging layout + idempotency
# ---------------------------------------------------------------------------


class TestBootstrapStaging:
    def test_train_staged_under_dataset_dir(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert (tmp_path / "dataset" / "core_train.json").is_file()

    def test_ciphertext_kept_pristine_under_dataset_dir(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert (tmp_path / "dataset" / "core_test.json.gpg").read_bytes() == _SENTINEL

    def test_decrypted_test_lands_at_raw_root(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert (tmp_path / "core_test.json").is_file()

    def test_decrypted_test_not_in_dataset_dir(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert not (tmp_path / "dataset" / "core_test.json").exists()

    def test_manifests_not_added_to_checksum_ledger(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert not (tmp_path / ".checksums.json").exists()

    def test_idempotent_second_run_stages_nothing(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Act
        with _swapped_seams(http, dec):
            result = _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert result["staged"] == {
            "train": False,
            "test_gpg": False,
            "test_decrypted": False,
        }

    def test_redecrypt_sets_test_decrypted_flag(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        (tmp_path / "core_test.json").unlink()
        # Act
        with _swapped_seams(http, dec):
            result = _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert result["staged"]["test_decrypted"] is True

    def test_redecrypt_does_not_refetch_ciphertext(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        (tmp_path / "core_test.json").unlink()
        before = http.n_gpg_fetches()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Assert
        assert http.n_gpg_fetches() == before

    def test_force_redoes_both_fetch_and_decrypt(self, tmp_path):
        # Arrange
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        with _swapped_seams(http, dec):
            _corebench_download.bootstrap_oracle(tmp_path)
        # Act
        with _swapped_seams(http, dec):
            result = _corebench_download.bootstrap_oracle(tmp_path, force=True)
        # Assert
        assert result["staged"] == {
            "train": True,
            "test_gpg": True,
            "test_decrypted": True,
        }


# ---------------------------------------------------------------------------
# _decrypt_gpg — real gpg round-trip (skipped when gpg absent)
# ---------------------------------------------------------------------------

_GPG = shutil.which("gpg") or shutil.which("gpg2")


def _symmetric_encrypt(plaintext: Path, dest: Path, passphrase: str) -> None:
    subprocess.run(
        [
            _GPG,
            "--batch",
            "--yes",
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            passphrase,
            "--cipher-algo",
            "AES256",
            "--symmetric",
            "--output",
            str(dest),
            str(plaintext),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(_GPG is None, reason="gpg binary not installed")
class TestRealGpgRoundTrip:
    def test_production_decrypt_recovers_plaintext(self, tmp_path):
        # Arrange
        plain = tmp_path / "secret.json"
        plain.write_text(json.dumps([_test_record()]))
        enc = tmp_path / "secret.json.gpg"
        _symmetric_encrypt(plain, enc, "reproducibility")
        out = tmp_path / "decrypted.json"
        # Act
        _corebench_download._decrypt_gpg(enc, out, "reproducibility")
        # Assert
        assert out.read_text() == plain.read_text()

    def test_wrong_passphrase_raises(self, tmp_path):
        # Arrange
        plain = tmp_path / "secret.json"
        plain.write_text(json.dumps([_test_record()]))
        enc = tmp_path / "secret.json.gpg"
        _symmetric_encrypt(plain, enc, "reproducibility")
        out = tmp_path / "decrypted.json"

        # Act
        def _run():
            _corebench_download._decrypt_gpg(enc, out, "wrong-pass")

        # Assert
        with pytest.raises(RuntimeError):
            _run()

    def test_wrong_passphrase_unlinks_partial_output(self, tmp_path):
        # Arrange
        plain = tmp_path / "secret.json"
        plain.write_text(json.dumps([_test_record()]))
        enc = tmp_path / "secret.json.gpg"
        _symmetric_encrypt(plain, enc, "reproducibility")
        out = tmp_path / "decrypted.json"
        # Act
        try:
            _corebench_download._decrypt_gpg(enc, out, "wrong-pass")
        except RuntimeError:
            pass
        # Assert
        assert not out.exists()


def test_decrypt_gpg_missing_binary_raises(tmp_path):
    # Arrange — force the which() lookup to miss.
    saved_which = shutil.which

    def _no_gpg(*_a, **_k):
        return None

    shutil.which = _no_gpg  # type: ignore[assignment]

    def _run():
        _corebench_download._decrypt_gpg(
            tmp_path / "x.gpg", tmp_path / "x.json", "reproducibility"
        )

    # Act
    raised = pytest.raises(RuntimeError, match="gpg not found")
    # Assert
    try:
        with raised:
            _run()
    finally:
        shutil.which = saved_which  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# download — bootstrap wiring + regressions
# ---------------------------------------------------------------------------


def _download_none_ids(raw):
    http = _UrlKeyedHttp()
    dec = _FakeDecrypt()
    with _swapped_seams(http, dec):
        result = _corebench_download.download(raw_dir=raw)
    return result, http, dec


class TestDownloadBootstrapWiring:
    def test_none_ids_stages_train_manifest(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        # Act
        _download_none_ids(raw)
        # Assert
        assert (raw / "dataset" / "core_train.json").is_file()

    def test_none_ids_stages_decrypted_test_manifest(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        # Act
        _download_none_ids(raw)
        # Assert
        assert (raw / "core_test.json").is_file()

    def test_none_ids_fetches_all_capsules(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        # Act
        result, _, _ = _download_none_ids(raw)
        # Assert
        assert result["n_fetched"] == 2

    def test_none_ids_result_carries_oracle_summary(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        # Act
        result, _, _ = _download_none_ids(raw)
        # Assert
        assert "oracle" in result

    def test_none_ids_uses_deduped_sorted_capsule_ids(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        # Act
        _, http, _ = _download_none_ids(raw)
        # Assert
        fetched = [
            u.rsplit("/", 1)[-1] for u, _ in http.calls if u.endswith(".tar.gz")
        ]
        assert fetched == ["capsule-1111111.tar.gz", "capsule-2222222.tar.gz"]

    def test_explicit_ids_no_upstream_fetch(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.download(raw_dir=raw, capsule_ids=["capsule-1111111"])
        # Assert
        assert all("githubusercontent" not in u for u, _ in http.calls)

    def test_explicit_ids_no_decrypt(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            _corebench_download.download(raw_dir=raw, capsule_ids=["capsule-1111111"])
        # Assert
        assert dec.calls == 0

    def test_explicit_ids_no_oracle_key_in_result(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"
        http, dec = _UrlKeyedHttp(), _FakeDecrypt()
        # Act
        with _swapped_seams(http, dec):
            result = _corebench_download.download(
                raw_dir=raw, capsule_ids=["capsule-1111111"]
            )
        # Assert
        assert "oracle" not in result

    def test_fetch_failure_counts_one(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"

        def _boom(url, dest):
            raise OSError("connection reset")

        # Act
        with _swap("_http_download", _boom):
            result = _corebench_download.download(
                raw_dir=raw, capsule_ids=["capsule-9999999"]
            )
        # Assert
        assert result["n_failed"] == 1

    def test_fetch_failure_records_failing_url(self, tmp_path):
        # Arrange
        raw = tmp_path / "raw"

        def _boom(url, dest):
            raise OSError("connection reset")

        # Act
        with _swap("_http_download", _boom):
            result = _corebench_download.download(
                raw_dir=raw, capsule_ids=["capsule-9999999"]
            )
        # Assert
        assert "capsule-9999999.tar.gz" in result["failures"][0]


class TestPrepareFromEmptyTree:
    def _paths(self, root):
        return _base.BenchmarkPaths(
            benchmark=corebench.BENCHMARK,
            root=root,
            raw_dir=root / "raw",
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
            manifest_dir=root / ".scitex" / "dataset",
        )

    def test_prepare_from_empty_tree_emits_manifest(self, tmp_path):
        # Arrange
        paths = self._paths(tmp_path)
        # Act
        with _swapped_seams(_UrlKeyedHttp(), _FakeDecrypt()):
            result = corebench.prepare(paths=paths, skip_download=False)
        # Assert
        assert Path(result["manifest"]).is_file()

    def test_prepare_from_empty_tree_emits_index_mapper(self, tmp_path):
        # Arrange
        paths = self._paths(tmp_path)
        # Act
        with _swapped_seams(_UrlKeyedHttp(), _FakeDecrypt()):
            corebench.prepare(paths=paths, skip_download=False)
        # Assert
        assert (paths.for_solver_dir / "index.jsonl").is_file()

    def test_prepare_from_empty_tree_emits_capsule_task_jsonl(self, tmp_path):
        # Arrange
        paths = self._paths(tmp_path)
        # Act
        with _swapped_seams(_UrlKeyedHttp(), _FakeDecrypt()):
            corebench.prepare(paths=paths, skip_download=False)
        # Assert
        assert (paths.for_solver_dir / "capsule-001" / "task.jsonl").is_file()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
