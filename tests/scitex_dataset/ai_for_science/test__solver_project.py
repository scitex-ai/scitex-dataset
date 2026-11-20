#!/usr/bin/env python3
"""Tests for the guarded solver-project assembly (``_solver_project``).

The capsule is prepared through the REAL ``bixbench.standardize`` path (a
genuine zip archive, a real answer manifest), so the project under test is
exactly what a harness would hand to an agent. No mocks, no monkeypatch;
the HF seam for the selective download is a hand-rolled stub module.
"""

import json
import shutil
import subprocess
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import bixbench
from scitex_dataset.ai_for_science._solver_project import (
    SolverProjectLeakError,
    assert_solver_project_clean,
    build_solver_project,
    eval_answer_values,
    prepare_solver_project,
    submission_from_answers,
)

ARCHIVE_A = "CapsuleFolder-aaaa1111-2222-3333-4444-555566667777.zip"
ARCHIVE_B = "CapsuleFolder-bbbb1111-2222-3333-4444-555566667777.zip"
# Distinctive reference values: a 5-significant-digit number, a range.
RECORDS = [
    {"question": "What is the odds ratio of the adjusted model?", "short_id": "bix-7",
     "question_id": "bix-7-q1", "data_folder": ARCHIVE_A, "answer": True, "ideal": "1.5242"},
    {"question": "Which group differs significantly?", "short_id": "bix-7",
     "question_id": "bix-7-q2", "data_folder": ARCHIVE_A, "answer": False, "ideal": "(0.0241,0.0262)"},
    {"question": "What is the mean expression?", "short_id": "bix-9",
     "question_id": "bix-9-q1", "data_folder": ARCHIVE_B, "answer": True, "ideal": "7.3381"},
]  # fmt: skip


def _zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "Folder/analysis.py",
            "import pandas as pd\nprint(pd.read_csv('data.csv').mean())\n",
        )
        zf.writestr("Folder/data.csv", "group,value\na,1.1\nb,2.2\n")


@pytest.fixture
def root(tmp_path):
    """A dataset root with bixbench prepared (both capsules materialized)."""
    raw = tmp_path / "ds" / "ai-for-science" / "bixbench" / "raw"
    raw.mkdir(parents=True)
    (raw / "BixBench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    _zip(raw / ARCHIVE_A)
    _zip(raw / ARCHIVE_B)
    base = raw.parent
    bixbench.standardize(
        raw_dir=raw, for_solver_dir=base / "for_solver", eval_dir=base / "eval"
    )
    return tmp_path / "ds"


@pytest.fixture
def built(root, tmp_path):
    base = root / "ai-for-science" / "bixbench"
    doc = build_solver_project(base / "for_solver" / "capsule-001", tmp_path / "p1",
                               mapping=tmp_path / "sealed" / "p1.json", eval_dir=base / "eval")  # fmt: skip
    return {"doc": doc, "project": tmp_path / "p1", "eval": base / "eval",
            "mapping": tmp_path / "sealed" / "p1.json"}  # fmt: skip


def _check(built):
    m = json.loads(built["mapping"].read_text())
    return assert_solver_project_clean(built["project"], eval_dir=built["eval"],
                                       task_ids=m["ids"].values(), native_id=m["native_id"],
                                       path_patterns=bixbench.PATH_ID_PATTERNS)  # fmt: skip


def _kinds(built):
    try:
        _check(built)
    except SolverProjectLeakError as exc:
        return {v["kind"] for v in exc.violations}
    return set()


def _text_of(project: Path) -> str:
    return "".join(
        f.read_text(errors="ignore") + str(f) for f in project.rglob("*") if f.is_file()
    )


# --- assembly --------------------------------------------------------------


def test_project_questions_are_numbered_prompts(built):
    # Arrange
    path = built["project"] / "questions.json"
    # Act
    questions = json.loads(path.read_text())
    # Assert
    assert questions == [
        {"id": "q1", "question": RECORDS[0]["question"]},
        {"id": "q2", "question": RECORDS[1]["question"]},
    ]


def test_project_holds_no_capsule_task_file(built):
    # Arrange
    project = built["project"]
    # Act
    names = {x.name for x in project.rglob("*")}
    # Assert
    assert not names & {"task.jsonl", "README.md", "submission.example.json"}


def test_build_reports_a_clean_guard(built):
    # Arrange
    doc = built["doc"]
    # Act
    clean = doc["guard"]["clean"]
    # Assert
    assert clean is True


def test_mapping_maps_question_ids_to_task_ids(built):
    # Arrange
    path = built["mapping"]
    # Act
    ids = json.loads(path.read_text())["ids"]
    # Assert
    assert ids == {"q1": "bixbench/bix-7-q1", "q2": "bixbench/bix-7-q2"}


@pytest.mark.parametrize("token", ["bix-7", "bixbench", "aaaa1111", "CapsuleFolder"])
def test_no_benchmark_or_capsule_identifier_reaches_the_project(built, token):
    # Arrange
    project = built["project"]
    # Act
    text = _text_of(project)
    # Assert
    assert token not in text


def test_refuses_mapping_inside_project(root, tmp_path):
    # Arrange
    base = root / "ai-for-science" / "bixbench"
    # Act
    call = lambda: build_solver_project(base / "for_solver" / "capsule-001", tmp_path / "p",  # noqa: E731
                                        mapping=tmp_path / "p" / "m.json", eval_dir=base / "eval")  # fmt: skip
    # Assert
    with pytest.raises(ValueError):
        call()


def test_refuses_existing_destination(root, tmp_path):
    # Arrange
    base = root / "ai-for-science" / "bixbench"
    (tmp_path / "p").mkdir()
    # Act
    call = lambda: build_solver_project(base / "for_solver" / "capsule-001", tmp_path / "p",  # noqa: E731
                                        mapping=tmp_path / "m.json", eval_dir=base / "eval")  # fmt: skip
    # Assert
    with pytest.raises(FileExistsError):
        call()


def test_prepare_solver_project_resolves_a_native_capsule_id(root, tmp_path):
    # Arrange
    native = ARCHIVE_B.removesuffix(".zip")
    # Act
    doc = prepare_solver_project("bixbench", native, tmp_path / "p2",
                                 mapping=tmp_path / "m2.json", dataset_root=root)  # fmt: skip
    # Assert
    assert doc["ids"] == {"q1": "bixbench/bix-9-q1"}


def test_prepare_solver_project_refuses_an_unmaterialized_capsule(root, tmp_path):
    # Arrange
    shutil.rmtree(root / "ai-for-science/bixbench/for_solver/capsule-002/input")
    # Act
    call = lambda: prepare_solver_project("bixbench", "capsule-002", tmp_path / "p",  # noqa: E731
                                          mapping=tmp_path / "m.json", dataset_root=root)  # fmt: skip
    # Assert
    with pytest.raises(FileNotFoundError):
        call()


# --- the guard: every planted leak class is refused ---------------------------


def test_clean_project_passes_the_guard(built):
    # Arrange
    project = built
    # Act
    kinds = _kinds(project)
    # Assert
    assert kinds == set()


@pytest.mark.parametrize(
    "plant, kind",
    [
        (lambda p, ev: (p / "answers.jsonl").write_text("{}\n"), "operator-file"),
        (lambda p, ev: (p / "evaluate.py").write_text("x = 1\n"), "operator-file"),
        (lambda p, ev: (p / "BixBench.jsonl").write_text("\n"), "operator-file"),
        (lambda p, ev: shutil.copy(ev / "answers.jsonl", p / "notes.dat"), "eval-copy"),
        (lambda p, ev: (p / "ref").symlink_to(ev), "symlink-outside"),
        (lambda p, ev: (p / "notes.md").write_text("see bix-7-q2\n"), "identifier-in-text"),
        (lambda p, ev: (p / "bix-7-extra.txt").write_text("x\n"), "identifier-in-path"),
        (lambda p, ev: (p / "summary.txt").write_text("OR = 1.524\n"), "answer-value"),
        (lambda p, ev: (p / "summary.txt").write_text("CI (0.0241,0.0262)\n"), "answer-value"),
    ],
    ids=["answers-jsonl", "evaluate-py", "upstream-manifest", "renamed-key", "symlink",
         "task-id-text", "task-id-path", "rounded-value", "range-value"],
)  # fmt: skip
def test_guard_refuses_each_planted_leak_class(built, plant, kind):
    # Arrange
    plant(built["project"], built["eval"])
    # Act
    kinds = _kinds(built)
    # Assert
    assert kind in kinds


def test_violation_messages_carry_no_answer_value(built):
    # Arrange
    (built["project"] / "summary.txt").write_text("OR = 1.5242\n")
    # Act
    try:
        _check(built)
        message = ""
    except SolverProjectLeakError as exc:
        message = str(exc) + json.dumps(exc.violations)
    # Assert
    assert message and "1.524" not in message


def test_failed_build_leaves_neither_project_nor_mapping(root, tmp_path):
    # Arrange
    base = root / "ai-for-science" / "bixbench"
    (base / "for_solver/capsule-001/input/leak.txt").write_text("ideal 1.5242\n")
    # Act
    try:
        build_solver_project(base / "for_solver" / "capsule-001", tmp_path / "p",
                             mapping=tmp_path / "m.json", eval_dir=base / "eval")  # fmt: skip
    except SolverProjectLeakError:
        pass
    # Assert
    assert [p.exists() for p in (tmp_path / "p", tmp_path / "m.json")] == [False, False]


def test_eval_answer_values_include_reference_string_and_number(built):
    # Arrange
    tid = "bixbench/bix-7-q1"
    # Act
    vals = eval_answer_values(built["eval"], [tid])
    # Assert
    assert {"1.5242", 1.5242} <= set(vals) and True not in vals


# --- submissions -------------------------------------------------------------


def test_submission_from_answers_maps_back_and_marks_unanswered_none(built, tmp_path):
    # Arrange
    ans = tmp_path / "answers.json"
    ans.write_text(json.dumps({"q2": "(0.02,0.03)"}))
    # Act
    sub = submission_from_answers(ans, built["mapping"])
    # Assert
    assert sub == [
        {"task_id": "bixbench/bix-7-q1", "answer": None},
        {"task_id": "bixbench/bix-7-q2", "answer": "(0.02,0.03)"},
    ]


def test_submission_from_missing_answers_is_all_none(built):
    # Arrange
    mapping = built["mapping"]
    # Act
    sub = submission_from_answers(None, mapping)
    # Assert
    assert [r["answer"] for r in sub] == [None, None]


# --- CLI ----------------------------------------------------------------------


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "scitex_dataset", *args],
                          capture_output=True, text=True, check=False)  # fmt: skip


def test_cli_solver_project_writes_a_project(root, tmp_path):
    # Arrange
    out = tmp_path / "c1"
    # Act
    res = _cli("ai-for-science", "bixbench", "build-solver-project", "capsule-001", str(out),
               "--mapping", str(tmp_path / "c1.json"), "--dataset-root", str(root), "--json")  # fmt: skip
    # Assert
    assert res.returncode == 0 and (out / "questions.json").is_file(), res.stderr


def test_cli_solver_project_refuses_with_exit_2_and_no_output(root, tmp_path):
    # Arrange
    (root / "ai-for-science/bixbench/for_solver/capsule-001/input/leak.txt").write_text(
        "1.5242\n"
    )
    out = tmp_path / "c2"
    # Act
    res = _cli("ai-for-science", "bixbench", "build-solver-project", "capsule-001", str(out),
               "--mapping", str(tmp_path / "c2.json"), "--dataset-root", str(root))  # fmt: skip
    # Assert
    assert (res.returncode, "answer-value" in res.stderr, out.exists()) == (
        2,
        True,
        False,
    )


def test_cli_solver_project_dry_run_writes_nothing(root, tmp_path):
    # Arrange
    out = tmp_path / "c3"
    # Act
    res = _cli("ai-for-science", "bixbench", "build-solver-project", "capsule-001", str(out),
               "--mapping", str(tmp_path / "c3.json"), "--dataset-root", str(root), "--dry-run", "-y")  # fmt: skip
    # Assert
    assert (res.returncode, out.exists(), (tmp_path / "c3.json").exists()) == (
        0,
        False,
        False,
    )


# --- selective BixBench acquisition -------------------------------------------


@contextmanager
def _hf(recorder):
    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = recorder  # type: ignore[attr-defined]
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = stub
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


class _Hub:
    """Serves files from a local 'upstream' dir, honouring allow_patterns."""

    def __init__(self, upstream: Path):
        self.upstream, self.calls = upstream, []

    def __call__(self, *, local_dir, allow_patterns=None, **_):
        self.calls.append(allow_patterns)
        for f in self.upstream.iterdir():
            if allow_patterns is None or f.name in allow_patterns:
                shutil.copy(f, Path(local_dir) / f.name)
        return local_dir


@pytest.mark.parametrize(
    "selector", ["bix-9", "capsule-002", ARCHIVE_B.removesuffix(".zip")]
)
def test_download_capsule_ids_fetches_only_manifest_and_selected_archive(
    root, tmp_path, selector
):
    # Arrange
    hub = _Hub(root / "ai-for-science/bixbench/raw")
    raw = tmp_path / "raw2"
    # Act
    with _hf(hub):
        bixbench.download(raw_dir=raw, capsule_ids=[selector])
    # Assert
    assert sorted(p.name for p in raw.iterdir()) == ["BixBench.jsonl", ARCHIVE_B]


def test_download_unknown_capsule_refuses(root, tmp_path):
    # Arrange
    hub = _Hub(root / "ai-for-science/bixbench/raw")
    # Act
    call = lambda: bixbench.download(raw_dir=tmp_path / "raw3", capsule_ids=["bix-404"])  # noqa: E731
    # Assert
    with _hf(hub), pytest.raises(KeyError):
        call()
