#!/usr/bin/env python3
"""Neutral path names in for_solver (#111) — no mocks, synthetic archives.

Every identifier below is INVENTED. The archives are built in ``tmp_path``
and mirror only the general shape described in issue #111: a top-level
folder and notebook/data/script names that embed the capsule UUID and a
``<prefix>-<n>`` benchmark id, with code that references those names.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import (
    _identifiers,
    _materialization_identity,
    _neutral_paths,
    _standardize,
    bixbench,
)

UUID = "0f0e0d0c-0b0a-4908-8706-050403020100"
NATIVE_BIX = f"CapsuleFolder-{UUID}"
BIX_ID = "bix-7"
NATIVE_CORE = "capsule-7654321"
DIGITS = "7654321"

# Runs every code cell of a notebook with the notebook's directory as cwd.
NOTEBOOK_RUNNER = """
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), {})
"""


def _notebook(code: str, metadata: dict | None = None) -> str:
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 3,
                    "id": "c1",
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "text": ["3\n"]}],
                    "source": [code],
                }
            ],
            "metadata": metadata or {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def _stage_bixbench(tmp_path: Path) -> Path:
    """A BixBench-shaped zip whose names embed the UUID and ``bix-7``."""
    raw = tmp_path / "bixbench" / "raw"
    raw.mkdir(parents=True)
    record = {
        "question": "How many genes are listed?",
        "short_id": BIX_ID,
        "question_id": f"{BIX_ID}-q1",
        "data_folder": f"{NATIVE_BIX}.zip",
        "answer": True,
        "ideal": "synthetic-free-text-ideal",
    }
    (raw / "BixBench.jsonl").write_text(json.dumps(record) + "\n")
    top = NATIVE_BIX
    data_rel = f"CapsuleData-{UUID}/counts_{BIX_ID}.csv"
    nb_name = f"{BIX_ID}_{UUID}.ipynb"
    script = (
        "from pathlib import Path\n"
        "here = Path(__file__).parent\n"
        f"rows = (here / 'CapsuleData-{UUID}' / 'counts_{BIX_ID}.csv')"
        ".read_text().splitlines()\n"
        f"assert (here / 'CapsuleNotebook-{UUID}' / '{nb_name}').is_file()\n"
        "print('script', len(rows))\n"
    )
    cell = f"print('notebook', len(open('../{data_rel}').read().splitlines()))\n"
    metadata = {
        # An escaped slash hides the DOI from a raw-text scan of the JSON.
        "source_doi_escaped": "10.5555\\/synthetic.0042",
        "capsule": NATIVE_BIX,
    }
    with zipfile.ZipFile(raw / record["data_folder"], "w") as zf:
        zf.writestr(
            "README.md", f"Run `python {top}/main_{BIX_ID}.py` from this folder.\n"
        )
        zf.writestr(f"{top}/main_{BIX_ID}.py", script)
        zf.writestr(f"{top}/{data_rel}", "gene,value\nA,1\nB,2\n")
        zf.writestr(
            f"{top}/CapsuleNotebook-{UUID}/{nb_name}",
            _notebook(cell, metadata).replace("10.5555\\\\/", "10.5555\\/"),
        )
    return raw


@pytest.fixture
def bix(tmp_path):
    raw = _stage_bixbench(tmp_path)
    root = raw.parent
    out = bixbench.standardize(
        raw_dir=raw, for_solver_dir=root / "for_solver", eval_dir=root / "eval"
    )
    capsule = root / "for_solver" / "capsule-001"
    return {"root": root, "capsule": capsule, "input": capsule / "input", "out": out}


def _all_names(root: Path) -> list[str]:
    names = []
    for parent, dirs, files in os.walk(root):
        names.extend(dirs + files)
    return names


def _all_text(root: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in root.rglob("*")
        if p.is_file()
    )


def _new_path(out: dict, suffix: str) -> str:
    renames = out["for_solver"]["path_renames"]["capsule-001"]["renames"]
    return next(r["new"] for r in renames if r["old"].endswith(suffix))


class TestBixBenchPathNames:
    def test_no_path_component_carries_uuid(self, bix):
        # Arrange
        names = _all_names(bix["capsule"])
        # Act
        leaking = [n for n in names if UUID.lower() in n.lower()]
        # Assert
        assert leaking == []

    def test_no_path_component_carries_benchmark_id(self, bix):
        # Arrange
        names = _all_names(bix["capsule"])
        # Act
        leaking = [n for n in names if BIX_ID in n.lower()]
        # Assert
        assert leaking == []

    def test_no_content_carries_uuid(self, bix):
        # Arrange
        text = _all_text(bix["capsule"])
        # Act
        present = UUID in text
        # Assert
        assert not present

    def test_no_content_carries_benchmark_id_reference(self, bix):
        # Arrange
        text = _all_text(bix["input"])
        # Act
        present = BIX_ID in text
        # Assert
        assert not present

    def test_extensions_kept_with_neutral_names(self, bix):
        # Arrange
        # Act
        new = _new_path(bix["out"], f"{BIX_ID}_{UUID}.ipynb")
        # Assert
        assert new == "folder_02/folder_03/notebook_01.ipynb"

    def test_renamed_script_still_runs(self, bix):
        # Arrange
        script = bix["input"] / _new_path(bix["out"], f"main_{BIX_ID}.py")
        # Act
        done = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Assert
        assert done.stdout.strip() == "script 3", done.stderr

    def test_renamed_notebook_still_runs(self, bix, tmp_path):
        # Arrange
        notebook = bix["input"] / _new_path(bix["out"], ".ipynb")
        runner = tmp_path / "runner" / "run_notebook.py"
        runner.parent.mkdir()
        runner.write_text(NOTEBOOK_RUNNER)
        # Act
        done = subprocess.run(
            [sys.executable, "-I", str(runner), str(notebook)],
            capture_output=True,
            text=True,
            cwd=notebook.parent,
            timeout=60,
        )
        # Assert
        assert done.stdout.strip() == "notebook 3", done.stderr

    def test_readme_reference_rewritten(self, bix):
        # Arrange
        script = _new_path(bix["out"], f"main_{BIX_ID}.py")
        # Act
        readme = (bix["input"] / "README.md").read_text()
        # Assert
        assert f"python {script}" in readme

    def test_mapping_recorded_on_eval_side(self, bix):
        # Arrange
        path = bix["root"] / "eval" / "path_renames.json"
        # Act
        mapping = json.loads(path.read_text())
        # Assert
        assert mapping["capsule-001"]["native_id"] == NATIVE_BIX

    def test_mapping_not_under_for_solver(self, bix):
        # Arrange
        # Act
        found = list((bix["root"] / "for_solver").rglob("path_renames.json"))
        # Assert
        assert found == []

    def test_escaped_notebook_metadata_doi_redacted(self, bix):
        # Arrange
        notebook = bix["input"] / _new_path(bix["out"], ".ipynb")
        # Act
        decoded = json.loads(notebook.read_text())["metadata"]["source_doi_escaped"]
        # Assert
        assert "10.5555" not in decoded

    def test_notebook_outputs_cleared(self, bix):
        # Arrange
        notebook = bix["input"] / _new_path(bix["out"], ".ipynb")
        # Act
        cell = json.loads(notebook.read_text())["cells"][0]
        # Assert
        assert cell["outputs"] == []


def _stage_core(tmp_path: Path) -> tuple[Path, list[dict]]:
    """A CORE-Bench-shaped tarball: wrapper dir + importable id-named module."""
    raw = tmp_path / "raw"
    src = tmp_path / "src" / NATIVE_CORE
    (src / "code").mkdir(parents=True)
    (src / "data").mkdir()
    (src / "code" / f"helpers_{DIGITS}.py").write_text(
        "def load(path):\n    return len(open(path).read().splitlines())\n"
    )
    (src / "code" / f"run_{NATIVE_CORE}.py").write_text(
        f"from helpers_{DIGITS} import load\n"
        f"print('rows', load('../data/{NATIVE_CORE}_values.csv'))\n"
    )
    (src / "data" / f"{NATIVE_CORE}_values.csv").write_text("x\n1\n2\n3\n")
    os.symlink(f"{NATIVE_CORE}_values.csv", src / "data" / "latest.csv")
    archive = raw / "capsules" / f"{NATIVE_CORE}.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname=NATIVE_CORE)
    tasks = [
        {
            "task_id": "fixture/q1",
            "benchmark": "fixture",
            "prompt": "How many rows?",
            "data": f"./capsules/{NATIVE_CORE}.tar.gz",
        }
    ]
    return raw, tasks


@pytest.fixture
def core(tmp_path):
    raw, tasks = _stage_core(tmp_path)
    out = _standardize.write_for_solver_per_capsule(
        for_solver_dir=tmp_path / "for_solver",
        tasks=tasks,
        raw_dir=raw,
        eval_dir=tmp_path / "eval",
    )
    return {"input": tmp_path / "for_solver" / "capsule-001" / "input", "out": out}


class TestDigitCoreAndImports:
    def test_digit_core_renamed(self, core):
        # Arrange
        names = _all_names(core["input"])
        # Act
        leaking = [n for n in names if DIGITS in n]
        # Assert
        assert leaking == []

    def test_renamed_import_still_runs(self, core):
        # Arrange
        script = core["input"] / _new_path(
            {"for_solver": core["out"]}, f"run_{NATIVE_CORE}.py"
        )
        # Act
        done = subprocess.run(
            # The script's own directory is sys.path[0], so the renamed
            # sibling module must be importable under its NEW name.
            [sys.executable, "-s", str(script)],
            capture_output=True,
            text=True,
            cwd=script.parent,
            timeout=60,
        )
        # Assert
        assert done.stdout.strip() == "rows 4", done.stderr

    def test_symlink_target_rewritten(self, core):
        # Arrange
        link = core["input"] / "data" / "latest.csv"
        # Act
        resolved = link.resolve().is_file()
        # Assert
        assert resolved and DIGITS not in os.readlink(link)


class TestPathGuard:
    def test_guard_refuses_uuid_named_notebook(self, tmp_path):
        # Arrange
        (tmp_path / f"{UUID}.ipynb").write_text("{}")
        # Act
        # Assert
        with pytest.raises(_neutral_paths.IdentifierPathLeakError):
            _neutral_paths.assert_no_identifier_paths(tmp_path, capsule="x")

    def test_guard_refuses_benchmark_id_dir(self, tmp_path):
        # Arrange
        (tmp_path / f"analysis_{BIX_ID}").mkdir()
        # Act
        # Assert
        with pytest.raises(_neutral_paths.IdentifierPathLeakError):
            _neutral_paths.assert_no_identifier_paths(
                tmp_path, patterns=bixbench.PATH_ID_PATTERNS, capsule="x"
            )

    def test_guard_refuses_exact_id_joined_by_underscore(self, tmp_path):
        # Arrange
        (tmp_path / f"{NATIVE_CORE}_notes.txt").write_text("")
        # Act
        # Assert
        with pytest.raises(_neutral_paths.IdentifierPathLeakError):
            _neutral_paths.assert_no_identifier_paths(
                tmp_path, [NATIVE_CORE], capsule="x"
            )

    def test_guard_passes_neutral_tree(self, tmp_path):
        # Arrange
        (tmp_path / "folder_01").mkdir()
        (tmp_path / "folder_01" / "notebook_01.ipynb").write_text("{}")
        # Act
        outcome = _neutral_paths.assert_no_identifier_paths(
            tmp_path, [NATIVE_CORE], bixbench.PATH_ID_PATTERNS, capsule="x"
        )
        # Assert
        assert outcome is None

    def test_neutral_name_skips_existing_name(self, tmp_path):
        # Arrange
        (tmp_path / "file_01.csv").write_text("")
        (tmp_path / f"{UUID}.csv").write_text("")
        # Act
        renames = _neutral_paths.neutralize_identifier_paths(tmp_path)
        # Assert
        assert renames == [{"old": f"{UUID}.csv", "new": "file_02.csv"}]


class TestNotebookStructuralScan:
    def test_guard_finds_escaped_doi_in_notebook_metadata(self, tmp_path):
        # Arrange
        (tmp_path / "nb.ipynb").write_text(
            '{"cells": [], "metadata": {"doi": "10.5555\\/synthetic.0042"}}'
        )
        # Act
        # Assert
        with pytest.raises(_identifiers.HostingIdentifierLeakError):
            _identifiers.assert_no_hosting_identifiers(tmp_path, capsule="x")

    def test_guard_finds_identifier_in_cell_metadata(self, tmp_path):
        # Arrange
        cells = [{"cell_type": "code", "metadata": {"src": f"capsule\\u002d{DIGITS}"}}]
        (tmp_path / "nb.ipynb").write_text(
            json.dumps({"cells": cells, "metadata": {}}).replace("\\\\u", "\\u")
        )
        # Act
        # Assert
        with pytest.raises(_identifiers.HostingIdentifierLeakError):
            _identifiers.assert_no_hosting_identifiers(
                tmp_path, [NATIVE_CORE], capsule="x"
            )


class TestContractVersion:
    def test_materialization_contract_is_v4(self):
        # Arrange
        # Act
        contract = _materialization_identity._MATERIALIZATION_CONTRACT
        # Assert
        assert contract == "per-capsule-source-identity-v4"

    def test_older_contract_output_refused(self, tmp_path):
        # Arrange
        raw, tasks = _stage_core(tmp_path)
        for_solver = tmp_path / "for_solver"
        _standardize.write_for_solver_per_capsule(
            for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
        )
        ledger = for_solver / ".materialization-identity.json"
        data = json.loads(ledger.read_text())
        data["contract"] = "per-capsule-source-identity-v3"
        ledger.write_text(json.dumps(data))
        # Act
        # Assert
        with pytest.raises(_standardize.StaleMaterializationError):
            _standardize.write_for_solver_per_capsule(
                for_solver_dir=for_solver, tasks=tasks, raw_dir=raw
            )

    def test_invalid_path_pattern_refused(self, tmp_path):
        # Arrange
        raw, tasks = _stage_core(tmp_path)
        # Act
        # Assert
        with pytest.raises(_standardize.InvalidReferenceSourceError):
            _standardize.write_for_solver_per_capsule(
                for_solver_dir=tmp_path / "for_solver",
                tasks=tasks,
                raw_dir=raw,
                path_patterns=["("],
            )
