#!/usr/bin/env python3
"""Tests for ai_for_science.bixbench mask + download (no network).

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``.
``snapshot_download`` is imported lazily inside ``download(...)`` from
``huggingface_hub``, so we swap a hand-rolled stub module into
``sys.modules['huggingface_hub']`` for the duration of the test.
"""

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import bixbench

# ---------------------------------------------------------------------------
# Network seam — hand-rolled huggingface_hub stub (no unittest.mock)
# ---------------------------------------------------------------------------


@contextmanager
def _swap_module(name: str, replacement):
    """Swap ``sys.modules[name]`` with ``replacement`` for the block."""
    saved = sys.modules.get(name)
    sys.modules[name] = replacement
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


class _SnapshotRecorder:
    """Records snapshot_download calls and writes a dummy file per pull."""

    def __init__(self, *, write_dummy=True):
        self.calls = []
        self._write_dummy = write_dummy

    def __call__(self, *, repo_id, repo_type, local_dir, max_workers, token):
        self.calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "local_dir": local_dir,
                "max_workers": max_workers,
                "token": token,
            }
        )
        if self._write_dummy:
            dst = Path(local_dir)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "BixBench.jsonl").write_text("{}\n")
        return str(local_dir)


def _hf_stub(recorder):
    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = recorder  # type: ignore[attr-defined]
    return stub


def _sample_record():
    return {
        "id": "rec-1",
        "tag": "Q",
        "version": "v1",
        "question": "Which gene rises?",
        "hypothesis": "Truncating ASXL1 alters expression.",
        "capsule_uuid": "abc-123",
        "short_id": "sh1",
        "question_id": "qid1",
        "categories": ["bio"],
        "data_folder": "CapsuleData-abc",
        "eval_mode": "open",
        "canary": "CANARY-SENTINEL",
        # Oracle fields that must get nulled:
        "answer": "ASXL1",
        "ideal": "gene X",
        "result": "p<0.01",
        "distractors": ["gene Y", "gene Z"],
        "paper": "https://doi.org/10.1234/abc",
    }


@pytest.fixture
def oracle_record():
    return _sample_record()


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Stage the upstream snapshot the way download(...) would leave raw/."""
    base = tmp_path / "ai-for-science" / "bixbench" / "raw"
    base.mkdir(parents=True)
    (base / "BixBench.jsonl").write_text(json.dumps(_sample_record()) + "\n")
    return base


class TestMaskRecord:
    def test_mask_record_nulls_answer(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["answer"] is None

    def test_mask_record_nulls_ideal(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["ideal"] is None

    def test_mask_record_nulls_result(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["result"] is None

    def test_mask_record_nulls_distractors(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["distractors"] is None

    def test_mask_record_nulls_paper(self, oracle_record):
        # Arrange
        rec = oracle_record
        # Act
        masked = bixbench.mask_record(rec)
        # Assert
        assert masked["paper"] is None

    def test_mask_record_preserves_hypothesis(self, oracle_record):
        # Arrange
        expected = oracle_record["hypothesis"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["hypothesis"] == expected

    def test_mask_record_preserves_canary(self, oracle_record):
        # Arrange
        expected = oracle_record["canary"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["canary"] == expected

    def test_mask_record_preserves_question(self, oracle_record):
        # Arrange
        expected = oracle_record["question"]
        # Act
        masked = bixbench.mask_record(oracle_record)
        # Assert
        assert masked["question"] == expected

    def test_mask_record_is_idempotent(self, oracle_record):
        # Arrange
        once = bixbench.mask_record(oracle_record)
        # Act
        twice = bixbench.mask_record(once)
        # Assert
        assert once == twice

    def test_mask_record_does_not_mutate_input(self, oracle_record):
        # Arrange
        snapshot = json.loads(json.dumps(oracle_record))
        # Act
        _ = bixbench.mask_record(oracle_record)
        # Assert
        assert oracle_record == snapshot


class TestMaskOnDisk:
    def test_mask_writes_questions_jsonl_in_masked_dir(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "questions.jsonl").is_file()

    def test_mask_records_count_matches_raw_line_count(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert result["n_records"] == 1

    def test_mask_creates_backward_compat_symlink(self, tmp_path, staged_raw_dir):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert (masked_dir / "BixBench_masked.jsonl").is_symlink()

    def test_mask_raises_when_raw_jsonl_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            bixbench.mask(raw_dir=bare, masked_dir=tmp_path / "out")


class TestMaskSymlinkView:
    def test_mask_symlinks_answer_free_capsule_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange — stage an answer-free capsule dir alongside the oracle.
        (staged_raw_dir / "CapsuleData-abc").mkdir()
        (staged_raw_dir / "CapsuleData-abc" / "data.csv").write_text("x,y\n1,2\n")
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        link = masked_dir / "CapsuleData-abc"
        assert link.is_symlink() and link.resolve() == (
            staged_raw_dir / "CapsuleData-abc"
        )

    def test_mask_does_not_symlink_oracle_manifest_into_masked_dir(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert not (masked_dir / "BixBench.jsonl").is_symlink()

    def test_mask_result_symlinked_list_is_non_empty(self, tmp_path, staged_raw_dir):
        # Arrange — answer-free content present so a link is created.
        (staged_raw_dir / "CapsuleData-abc").mkdir()
        masked_dir = staged_raw_dir.parent / "masked"
        # Act
        result = bixbench.mask(raw_dir=staged_raw_dir, masked_dir=masked_dir)
        # Assert
        assert len(result["symlinked"]) >= 1


class TestDownload:
    def test_download_returns_repo_id_in_pulled_list(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.download(raw_dir=raw_dir)
        # Assert
        assert result["snapshots_pulled"] == [bixbench.HF_REPO_ID]

    def test_download_calls_snapshot_download_once(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            bixbench.download(raw_dir=raw_dir)
        # Assert
        assert len(rec.calls) == 1

    def test_download_returns_raw_dir(self, tmp_path):
        # Arrange
        rec = _SnapshotRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.download(raw_dir=raw_dir)
        # Assert
        assert result["raw_dir"] == str(raw_dir)


class TestPrepareWithDownload:
    def test_prepare_with_download_emits_download_mask_manifest_keys(
        self, tmp_path, staged_raw_dir
    ):
        # Arrange — JSONL already staged; snapshot_download is a no-op so
        # the staged raw_dir survives the download step.
        from scitex_dataset.ai_for_science import _base

        rec = _SnapshotRecorder(write_dummy=False)
        root = staged_raw_dir.parent
        paths = _base.BenchmarkPaths(
            benchmark=bixbench.BENCHMARK,
            root=root,
            raw_dir=staged_raw_dir,
            masked_dir=root / "masked",
            manifest_dir=root / ".scitex" / "dataset",
        )
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.prepare(paths=paths, skip_download=False)
        # Assert
        assert {"download", "mask", "manifest"} <= set(result)

    def test_prepare_with_download_writes_manifest_file(self, tmp_path, staged_raw_dir):
        # Arrange
        from scitex_dataset.ai_for_science import _base

        rec = _SnapshotRecorder(write_dummy=False)
        root = staged_raw_dir.parent
        paths = _base.BenchmarkPaths(
            benchmark=bixbench.BENCHMARK,
            root=root,
            raw_dir=staged_raw_dir,
            masked_dir=root / "masked",
            manifest_dir=root / ".scitex" / "dataset",
        )
        # Act
        with _swap_module("huggingface_hub", _hf_stub(rec)):
            result = bixbench.prepare(paths=paths, skip_download=False)
        # Assert
        assert Path(result["manifest"]).is_file()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
