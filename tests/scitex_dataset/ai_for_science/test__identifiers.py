#!/usr/bin/env python3
"""Hosting-identifier redaction in for_solver (no mocks, real fixture files).

The fixture tree ``fixtures/hosting_identifiers/capsule-7654321`` mirrors the
shape of an exported Code Ocean capsule as found in prepared CORE-Bench
projects (``REPRODUCING.md``, ``environment/Dockerfile``,
``metadata/metadata.yml``, ``code/``). Every identifier in it is INVENTED.
Each test checks ONE identifier class is neutralised, or ONE piece of
reproduction information is kept.
"""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import (
    _identifiers,
    _standardize,
    bixbench,
    corebench,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hosting_identifiers" / "capsule-7654321"
NATIVE = "capsule-7654321"
TITLE = "Invented Fixture Method: Graph Classification by Link Ranking"
DOI_URL = "https://doi.org/10.5555/fixture.2024.0042"
PUBLISHED_UUID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"


def _exact() -> list[str]:
    return _identifiers.native_identifiers(
        NATIVE
    ) + _identifiers.publication_identifiers(DOI_URL, TITLE)


def _tar_fixture(dest: Path) -> None:
    """Pack the fixture tree as a real capsule tarball (modes preserved)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(FIXTURE, arcname=NATIVE)


@pytest.fixture(scope="module")
def redacted(tmp_path_factory):
    """A copy of the fixture tree after :func:`mask_hosting_identifiers`."""
    root = tmp_path_factory.mktemp("ids") / NATIVE
    shutil.copytree(FIXTURE, root)
    report = _identifiers.mask_hosting_identifiers(root, _exact())
    return {"root": root, "report": report}


def _read(redacted, rel: str) -> str:
    return (redacted["root"] / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Identifier classes — each must be gone after redaction
# ---------------------------------------------------------------------------


class TestIdentifierClassesRemoved:
    def test_published_image_uuid_removed(self, redacted):
        # Arrange
        text = _read(redacted, "REPRODUCING.md")
        # Act
        present = PUBLISHED_UUID in text
        # Assert
        assert not present

    def test_environment_digest_header_removed(self, redacted):
        # Arrange
        text = _read(redacted, "environment/Dockerfile")
        # Act
        present = "hash:sha256" in text
        # Assert
        assert not present

    def test_hosting_platform_links_removed(self, redacted):
        # Arrange
        text = _read(redacted, "REPRODUCING.md")
        # Act
        present = "https://codeocean.com" in text or "help.codeocean.com" in text
        # Assert
        assert not present

    def test_capsule_page_url_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "codeocean.com/capsule" in text
        # Assert
        assert not present

    def test_archive_result_file_url_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "zenodo.org" in text
        # Assert
        assert not present

    def test_preprint_url_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "arxiv.org" in text
        # Assert
        assert not present

    def test_markdown_result_image_link_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "accuracy.png" in text
        # Assert
        assert not present

    def test_html_result_image_link_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "confusion.png" in text
        # Assert
        assert not present

    def test_results_path_url_removed(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        present = "results/table2.html" in text
        # Assert
        assert not present

    def test_paper_doi_removed_from_code(self, redacted):
        # Arrange
        text = _read(redacted, "code/main.py")
        # Act
        present = "10.5555/fixture" in text
        # Assert
        assert not present

    def test_paper_doi_removed_from_metadata(self, redacted):
        # Arrange
        text = _read(redacted, "metadata/metadata.yml")
        # Act
        present = "10.5555" in text
        # Assert
        assert not present

    def test_paper_title_removed_from_metadata(self, redacted):
        # Arrange
        text = _read(redacted, "metadata/metadata.yml").lower()
        # Act
        present = "graph classification by link ranking" in text
        # Assert
        assert not present

    def test_native_capsule_id_removed_from_extensionless_script(self, redacted):
        # Arrange
        text = _read(redacted, "code/run")
        # Act
        present = "7654321" in text
        # Assert
        assert not present

    def test_report_names_every_identifier_class(self, redacted):
        # Arrange
        expected = {
            "published_image",
            "environment_digest",
            "hosting_url",
            "result_link",
            "doi",
            "exact_identifier",
        }
        # Act
        found = set(redacted["report"]["classes"])
        # Assert
        assert found == expected


# ---------------------------------------------------------------------------
# Reproduction information — must survive redaction, still usable
# ---------------------------------------------------------------------------


class TestReproductionInstructionsKept:
    def test_docker_run_command_names_buildable_local_image(self, redacted):
        # Arrange
        text = _read(redacted, "REPRODUCING.md")
        # Act
        usable = f"  {_identifiers.LOCAL_ENV_IMAGE} bash run" in text
        # Assert
        assert usable

    def test_docker_run_volume_mounts_kept(self, redacted):
        # Arrange
        text = _read(redacted, "REPRODUCING.md")
        # Act
        kept = '--volume "$PWD/code":/code' in text
        # Assert
        assert kept

    def test_base_image_from_line_kept(self, redacted):
        # Arrange
        text = _read(redacted, "environment/Dockerfile")
        # Act
        first_line = text.splitlines()[0]
        # Assert
        assert first_line == (
            "FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04"
        )

    def test_dockerfile_package_versions_kept(self, redacted):
        # Arrange
        original = (FIXTURE / "environment" / "Dockerfile").read_text().splitlines()[1:]
        # Act
        after = _read(redacted, "environment/Dockerfile").splitlines()
        # Assert
        assert after == original

    def test_requirements_file_unchanged(self, redacted):
        # Arrange
        original = (FIXTURE / "code" / "requirements.txt").read_text()
        # Act
        after = _read(redacted, "code/requirements.txt")
        # Assert
        assert after == original

    def test_code_repository_url_kept(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        kept = "https://github.com/example-lab/link-ranking\n" in text
        # Assert
        assert kept

    def test_tool_documentation_url_kept(self, redacted):
        # Arrange
        text = _read(redacted, "code/README.md")
        # Act
        kept = "https://pip.pypa.io/en/stable/" in text
        # Assert
        assert kept

    def test_run_script_command_kept(self, redacted):
        # Arrange
        text = _read(redacted, "code/run")
        # Act
        kept = 'python -u main.py --data ../data/wine.txt --out ../results "$@"' in text
        # Assert
        assert kept

    def test_run_script_stays_executable(self, redacted):
        # Arrange
        run = redacted["root"] / "code" / "run"
        # Act
        mode = run.stat().st_mode
        # Assert
        assert mode & stat.S_IXUSR

    def test_redacted_python_still_compiles(self, redacted, tmp_path):
        # Arrange
        source = redacted["root"] / "code" / "main.py"
        # Act
        compiled = py_compile.compile(
            str(source), cfile=str(tmp_path / "main.pyc"), doraise=True
        )
        # Assert
        assert Path(compiled).is_file()

    def test_support_email_kept(self, redacted):
        # Arrange
        text = _read(redacted, "REPRODUCING.md")
        # Act
        kept = "mailto:support@codeocean.com" in text
        # Assert
        assert kept


# ---------------------------------------------------------------------------
# Text-level rules and file handling
# ---------------------------------------------------------------------------


class TestRedactTextRules:
    def test_bare_digest_redacted(self):
        # Arrange
        text = "image digest sha256:" + "ab" * 32 + "\n"
        # Act
        new, _ = _identifiers.redact_text(text)
        # Assert
        assert new == "image digest [redacted-digest]\n"

    def test_pinned_base_image_digest_kept(self):
        # Arrange
        text = "FROM python:3.11@sha256:" + "cd" * 32 + "\n"
        # Act
        new, _ = _identifiers.redact_text(text)
        # Assert
        assert new == text

    def test_trailing_sentence_punctuation_kept(self):
        # Arrange
        text = "See https://zenodo.org/records/1."
        # Act
        new, _ = _identifiers.redact_text(text)
        # Assert
        assert new == "See [redacted-url]."

    def test_title_matched_across_line_wrap(self):
        # Arrange
        text = "Invented Fixture Method:\n  Graph Classification by Link Ranking"
        # Act
        new, _ = _identifiers.redact_text(text, [TITLE])
        # Assert
        assert new == "[redacted-identifier]"

    def test_unversioned_doi_also_listed(self):
        # Arrange
        doi = "https://doi.org/10.24433/CO.1234567.v1"
        # Act
        forms = _identifiers.publication_identifiers(doi)
        # Assert
        assert forms == ["10.24433/CO.1234567.v1", "10.24433/CO.1234567"]


class TestFileHandling:
    def test_read_only_member_redacted(self, tmp_path):
        # Arrange
        target = tmp_path / "REPRODUCING.md"
        target.write_text("see https://codeocean.com/capsule/1\n")
        target.chmod(0o444)
        # Act
        _identifiers.mask_hosting_identifiers(tmp_path)
        # Assert
        assert target.read_text() == "see [redacted-url]\n"

    def test_read_only_mode_restored(self, tmp_path):
        # Arrange
        target = tmp_path / "REPRODUCING.md"
        target.write_text("see https://codeocean.com/capsule/1\n")
        target.chmod(0o444)
        # Act
        _identifiers.mask_hosting_identifiers(tmp_path)
        # Assert
        assert stat.S_IMODE(target.stat().st_mode) == 0o444

    def test_binary_file_untouched(self, tmp_path):
        # Arrange
        blob = b"\x00\x01https://codeocean.com/capsule/1\x00"
        target = tmp_path / "weights.bin"
        target.write_bytes(blob)
        # Act
        _identifiers.mask_hosting_identifiers(tmp_path)
        # Assert
        assert target.read_bytes() == blob

    def test_symlink_target_outside_input_not_rewritten(self, tmp_path):
        # Arrange
        outside = tmp_path / "outside.md"
        outside.write_text("https://codeocean.com/capsule/1\n")
        inside = tmp_path / "input"
        inside.mkdir()
        os.symlink(outside, inside / "link.md")
        # Act
        _identifiers.mask_hosting_identifiers(inside)
        # Assert
        assert outside.read_text() == "https://codeocean.com/capsule/1\n"

    def test_non_utf8_bytes_preserved(self, tmp_path):
        # Arrange
        target = tmp_path / "notes.txt"
        target.write_bytes(b"caf\xe9 https://zenodo.org/records/9\n")
        # Act
        _identifiers.mask_hosting_identifiers(tmp_path)
        # Assert
        assert target.read_bytes() == b"caf\xe9 [redacted-url]\n"


class TestFailLoudGuard:
    def test_guard_raises_on_unredacted_capsule(self, tmp_path):
        # Arrange
        root = tmp_path / NATIVE
        shutil.copytree(FIXTURE, root)
        # Act
        # Assert
        with pytest.raises(_identifiers.HostingIdentifierLeakError):
            _identifiers.assert_no_hosting_identifiers(root, _exact(), capsule=NATIVE)

    def test_guard_passes_after_redaction(self, redacted):
        # Arrange
        root = redacted["root"]
        # Act
        outcome = _identifiers.assert_no_hosting_identifiers(
            root, _exact(), capsule=NATIVE
        )
        # Assert
        assert outcome is None


# ---------------------------------------------------------------------------
# Wired into the shared per-capsule writer and the adapters
# ---------------------------------------------------------------------------


def _materialize(tmp_path: Path, identifiers=None):
    raw = tmp_path / "raw"
    _tar_fixture(raw / "capsules" / f"{NATIVE}.tar.gz")
    tasks = [
        {
            "task_id": "fixture/q1",
            "benchmark": "fixture",
            "prompt": "Report the accuracy.",
            "data": f"./capsules/{NATIVE}.tar.gz",
        }
    ]
    for_solver = tmp_path / "for_solver"
    report = _standardize.write_for_solver_per_capsule(
        for_solver_dir=for_solver,
        tasks=tasks,
        raw_dir=raw,
        identifiers=identifiers,
    )
    return for_solver, raw, tasks, report


class TestPerCapsuleWriter:
    def test_writer_redacts_extracted_input(self, tmp_path):
        # Arrange
        for_solver, *_ = _materialize(tmp_path, {NATIVE: [TITLE, DOI_URL]})
        # Act
        text = (for_solver / "capsule-001" / "input" / "REPRODUCING.md").read_text()
        # Assert
        assert PUBLISHED_UUID not in text

    def test_writer_reports_identifier_classes(self, tmp_path):
        # Arrange
        # Act
        *_, report = _materialize(tmp_path, {NATIVE: [TITLE]})
        # Assert
        assert "published_image" in report["identifier_classes"]

    def test_capsule_readme_explains_local_environment_build(self, tmp_path):
        # Arrange
        for_solver, *_ = _materialize(tmp_path)
        # Act
        readme = (for_solver / "capsule-001" / "README.md").read_text()
        # Assert
        assert "docker build -t capsule-environment:local input/environment" in readme

    def test_changed_identifiers_refuse_cached_reuse(self, tmp_path):
        # Arrange
        for_solver, raw, tasks, _ = _materialize(tmp_path, {NATIVE: [TITLE]})
        # Act
        # Assert
        with pytest.raises(_standardize.StaleMaterializationError):
            _standardize.write_for_solver_per_capsule(
                for_solver_dir=for_solver,
                tasks=tasks,
                raw_dir=raw,
                identifiers={NATIVE: [TITLE, "Another Distinct Title"]},
            )

    def test_malformed_identifiers_refused(self, tmp_path):
        # Arrange
        # Act
        # Assert
        with pytest.raises(_standardize.InvalidReferenceSourceError):
            _materialize(tmp_path, {NATIVE: TITLE})


def _stage_corebench(tmp_path: Path) -> Path:
    raw = tmp_path / "corebench" / "raw"
    (raw / "dataset").mkdir(parents=True)
    record = {
        "capsule_id": NATIVE,
        "language": "Python",
        "field": "computer science",
        "task_prompt": "Run the capsule.",
        "capsule_title": TITLE,
        "capsule_doi": DOI_URL,
        "results": [{"What is the reported score?": 0.5}],
    }
    (raw / "dataset" / "core_train.json").write_text(json.dumps([record]))
    (raw / "core_test.json").write_text("[]")
    _tar_fixture(raw / "capsules" / f"{NATIVE}.tar.gz")
    return raw


class TestCoreBenchAdapter:
    def test_corebench_redacts_oracle_title_from_input(self, tmp_path):
        # Arrange
        raw = _stage_corebench(tmp_path)
        root = raw.parent
        corebench.standardize(
            raw_dir=raw, for_solver_dir=root / "for_solver", eval_dir=root / "eval"
        )
        # Act
        text = (
            root / "for_solver" / "capsule-001" / "input" / "metadata" / "metadata.yml"
        ).read_text()
        # Assert
        assert "Link Ranking" not in text


def _stage_bixbench(tmp_path: Path, ideal: str) -> Path:
    raw = tmp_path / "bixbench" / "raw"
    raw.mkdir(parents=True)
    record = {
        "question": "What is the adjusted p-value?",
        "short_id": "bix-9",
        "question_id": "bix-9-q1",
        "data_folder": "CapsuleFolder-0f0e0d0c-0b0a-4908-8706-050403020100.zip",
        "answer": True,
        "ideal": ideal,
    }
    (raw / "BixBench.jsonl").write_text(json.dumps(record) + "\n")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["Result: padj = 0.000213, gene set Upregulated in mutant"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with zipfile.ZipFile(raw / record["data_folder"], "w") as zf:
        zf.writestr("CapsuleNotebook/analysis.ipynb", json.dumps(notebook))
        zf.writestr("CapsuleData/counts.csv", "gene,padj\nA,0.000213\nB,0.5\n")
    return raw


def _bix_input(raw: Path) -> Path:
    root = raw.parent
    bixbench.standardize(
        raw_dir=raw, for_solver_dir=root / "for_solver", eval_dir=root / "eval"
    )
    return root / "for_solver" / "capsule-001" / "input"


class TestBixBenchAnswerRedaction:
    def test_numeric_string_answer_redacted_from_data(self, tmp_path):
        # Arrange
        input_dir = _bix_input(_stage_bixbench(tmp_path, "0.000213"))
        # Act
        text = (input_dir / "CapsuleData" / "counts.csv").read_text()
        # Assert
        assert "0.000213" not in text

    def test_distinctive_string_answer_redacted_from_notebook(self, tmp_path):
        # Arrange
        input_dir = _bix_input(_stage_bixbench(tmp_path, "Upregulated in mutant"))
        # Act
        text = (input_dir / "CapsuleNotebook" / "analysis.ipynb").read_text()
        # Assert
        assert "Upregulated in mutant" not in text

    def test_non_answer_values_kept(self, tmp_path):
        # Arrange
        input_dir = _bix_input(_stage_bixbench(tmp_path, "0.000213"))
        # Act
        text = (input_dir / "CapsuleData" / "counts.csv").read_text()
        # Assert
        assert "B,0.5" in text

    def test_capsule_uuid_redacted_from_input(self, tmp_path):
        # Arrange
        raw = _stage_bixbench(tmp_path, "0.000213")
        uuid = "0f0e0d0c-0b0a-4908-8706-050403020100"
        with zipfile.ZipFile(raw / f"CapsuleFolder-{uuid}.zip", "a") as zf:
            zf.writestr("CapsuleNotebook/README.txt", f"Capsule {uuid}\n")
        input_dir = _bix_input(raw)
        # Act
        text = (input_dir / "CapsuleNotebook" / "README.txt").read_text()
        # Assert
        assert uuid not in text
