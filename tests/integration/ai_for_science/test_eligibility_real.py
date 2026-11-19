#!/usr/bin/env python3
"""Eligibility screening on REAL upstream data (small selective downloads).

Each benchmark's own selective ``download`` stages a handful of the
smallest real units into a session temp dir (CORE-Bench: the oracle
manifests + seven capsules, ~5 MB; BixBench: the manifest + three
archives at a pinned revision, ~0.8 MB; BioMysteryBench: the 11 MB public
preview). No network -> the module skips with the reason printed; nothing
is substituted for the real data.

The expected facts are what those archives visibly contain (their
``environment/Dockerfile``, ``code/run`` and notebook metadata), and the
expected verdicts are the ones recorded when these units were screened by
hand for the first capsule batches.
"""

from __future__ import annotations

import json
import urllib.error

import pytest
from click.testing import CliRunner

from scitex_dataset.ai_for_science import biomysterybench, bixbench
from scitex_dataset.ai_for_science._corebench_download import (
    bootstrap_oracle,
)
from scitex_dataset.ai_for_science._corebench_download import (
    download as corebench_download,
)
from scitex_dataset.ai_for_science._eligibility import (
    ADAPTERS,
    ArchiveUnavailableError,
    screen_eligibility,
)

BIX_REVISION = "f8cc3bdcc6357c88b8c3648306522b9c422dc95a"
CB_CAPSULES = ["capsule-4098236", "capsule-6460826", "capsule-0940461",
               "capsule-2011424", "capsule-5975162", "capsule-5286757",
               "capsule-3894632"]  # fmt: skip
BIX_UNITS = ["bix-10", "bix-51", "bix-54"]


def _staged(fn):
    try:
        return fn()
    except FileNotFoundError:
        raise  # a missing staged file is a finding, never a skip
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"real upstream data unreachable: {exc}")


@pytest.fixture(scope="module")
def cb_raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("corebench") / "raw"

    def stage():
        bootstrap_oracle(raw)
        res = corebench_download(raw_dir=raw, capsule_ids=CB_CAPSULES)
        if res["n_failed"]:
            raise OSError(res["failures"])
        return raw

    return _staged(stage)


@pytest.fixture(scope="module")
def bix_raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("bixbench") / "raw"
    return _staged(lambda: bixbench.download(raw_dir=raw, capsule_ids=BIX_UNITS,
                                             revision=BIX_REVISION) and raw)  # fmt: skip


@pytest.fixture(scope="module")
def bmb_raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("biomysterybench") / "raw"
    return _staged(lambda: biomysterybench.download(raw_dir=raw) and raw)


def _facts(bench, raw, label):
    adapter = ADAPTERS[bench](raw, {})
    unit = next(u for u in adapter.units() if u["label"] == label)
    return adapter.archive_facts(unit, fetch=False)


# --- facts from real archives ------------------------------------------------


def test_corebench_three_script_run_counts_three_python_steps(cb_raw):
    # Arrange
    label = "capsule-6460826"  # run.sh: python iris / zoo / wine samples
    # Act
    facts = _facts("corebench", cb_raw, label)
    # Assert
    assert (facts["entry_kinds"], facts["n_entry_steps"]) == (["python"], 3)


def test_corebench_notebook_entry_records_its_local_module_imports(cb_raw):
    # Arrange
    label = (
        "capsule-0940461"  # run: nbconvert --execute; notebook imports 2 own modules
    )
    # Act
    facts = _facts("corebench", cb_raw, label)
    # Assert
    assert (facts["entry_kinds"], facts["entry_notebook_local_imports"]) == (
        ["notebook"],
        2,
    )


def test_corebench_cuda_base_image_is_gpu_in_environment(cb_raw):
    # Arrange
    label = "capsule-5286757"  # FROM ...miniconda3:4.7.10-cuda10.1-cudnn7...; torch
    # Act
    facts = _facts("corebench", cb_raw, label)
    # Assert
    assert (facts["gpu_in_environment"], facts["dl_frameworks"]) == (True, ["torch"])


def test_corebench_cpu_image_with_dl_wheels_lists_every_framework(cb_raw):
    # Arrange
    label = "capsule-5975162"  # CPU image; pip keras, tensorflow, torch
    # Act
    facts = _facts("corebench", cb_raw, label)
    # Assert
    assert (facts["gpu_in_environment"], facts["dl_frameworks"]) == (
        False, ["keras", "tensorflow", "torch"])  # fmt: skip


def test_corebench_rscript_entry_and_r_base_make_r_a_code_language(cb_raw):
    # Arrange
    label = "capsule-3894632"  # run: Rscript rmarkdown + python; conda r-base
    # Act
    facts = _facts("corebench", cb_raw, label)
    # Assert
    assert (facts["code_languages"], facts["entry_kinds"]) == (
        ["python", "r"],
        ["python", "r"],
    )


def test_bixbench_r_kernel_notebook_is_r_and_python_kernel_is_python(bix_raw):
    # Arrange
    labels = ("bix-51", "bix-10")
    # Act
    kernels = [
        _facts("bixbench", bix_raw, label)["kernel_languages"] for label in labels
    ]
    # Assert
    assert kernels == [["r"], ["python"]]


def test_bixbench_python_notebook_loading_rpy2_reports_r_as_magic_language(bix_raw):
    # Arrange
    label = "bix-10"  # Python kernel; a cell runs %load_ext rpy2.ipython
    # Act
    facts = _facts("bixbench", bix_raw, label)
    # Assert
    assert (facts["kernel_languages"], facts["magic_languages"]) == (["python"], ["r"])


def test_biomysterybench_preview_problem_ships_no_code(bmb_raw):
    # Arrange
    label = "hb002"
    # Act
    facts = _facts("biomysterybench", bmb_raw, label)
    # Assert
    assert (facts["entry_kinds"], facts["code_languages"]) == ([], [])


# --- end-to-end screening -----------------------------------------------------

CB_CONFIG = {
    "benchmark": "corebench",
    "rules": [
        {"id": "split", "fact": "split", "in": ["train"]},
        {"id": "language", "fact": "declared_language", "in": ["Python"]},
        {"id": "text_answers", "fact": "n_figure_questions", "max": 0},
        {"id": "size", "fact": "archive_bytes", "max": 1_000_000},
        {"id": "cpu_only", "fact": "gpu_in_environment", "equals": False},
        {"id": "light_deps", "all": [
            {"fact": "dl_frameworks", "none_of": ["tensorflow", "torch", "keras", "jax"]},
            {"fact": "code_languages", "none_of": ["r"]}]},
        {"id": "script_entry", "fact": "entry_kinds", "any_of": ["python"]},
        {"id": "multi_step", "fact": "n_entry_steps", "min": 2},
    ],
    "order": {"keys": [{"fact": "archive_bytes"}]},
    "target": None,
}  # fmt: skip


@pytest.fixture(scope="module")
def cb_result(cb_raw):
    return _staged(lambda: screen_eligibility(CB_CONFIG, raw_dir=cb_raw, fetch=False))


def test_corebench_screen_reasons_match_the_recorded_batch1_verdicts(cb_result):
    # Arrange
    recorded = {"capsule-4098236": "multi_step", "capsule-6460826": None,
                "capsule-0940461": "script_entry", "capsule-2011424": "multi_step",
                "capsule-5975162": "light_deps", "capsule-5286757": "cpu_only"}  # fmt: skip
    # Act
    got = {u["unit_id"]: (u["reason"] or {}).get("rule") for u in cb_result["units"]
           if u["position"] is not None}  # fmt: skip
    # Assert
    assert got == recorded


def test_corebench_screen_orders_candidates_by_archive_size(cb_result):
    # Arrange
    ranked = [u for u in cb_result["units"] if u["position"] is not None]
    # Act
    sizes = [u["facts"]["archive_bytes"] for u in ranked]
    # Assert
    assert sizes == sorted(sizes)


def test_corebench_screen_gives_every_unit_a_verdict(cb_result):
    # Arrange
    units = cb_result["units"]
    # Act
    missing = [u["unit_id"] for u in units if u["verdict"] is None]
    # Assert
    assert (len(units), missing) == (90, [])


def test_corebench_screen_is_deterministic(cb_raw, cb_result):
    # Arrange
    first = json.dumps(cb_result, sort_keys=True, default=str)
    # Act
    again = json.dumps(_staged(lambda: screen_eligibility(CB_CONFIG, raw_dir=cb_raw)),
                       sort_keys=True, default=str)  # fmt: skip
    # Assert
    assert again == first


def test_missing_archive_without_fetch_is_refused_not_skipped(cb_raw):
    # Arrange: admit a staged-but-absent candidate (capsule-1854976, 11.8 MB)
    cfg = dict(CB_CONFIG, rules=[{"id": "only", "fact": "unit_id", "in": ["capsule-1854976"]},
                                 {"id": "cpu", "fact": "gpu_in_environment", "equals": False}])  # fmt: skip
    # Act
    try:
        screen_eligibility(cfg, raw_dir=cb_raw, fetch=False)
        error = None
    except ArchiveUnavailableError as exc:
        error = exc
    # Assert
    assert isinstance(error, ArchiveUnavailableError)


def test_bixbench_screen_keeps_python_and_excludes_r_kernels(bix_raw):
    # Arrange
    cfg = {"benchmark": "bixbench", "source": {"revision": BIX_REVISION},
           "rules": [{"id": "B5", "fact": "n_questions", "min": 7},
                     {"id": "B6", "fact": "archive_bytes", "max": 1_000_000},
                     {"id": "B3", "all": [{"fact": "kernel_languages", "subset_of": ["python"]},
                                          {"fact": "file_languages", "none_of": ["r"]}]}],
           "order": {"keys": [{"fact": "n_questions", "descending": True},
                              {"fact": "archive_bytes", "descending": True}]}}  # fmt: skip
    # Act
    result = _staged(lambda: screen_eligibility(cfg, raw_dir=bix_raw, fetch=False))
    # Assert
    assert [(u["label"], u["verdict"]) for u in result["units"] if u["position"]] == [
        ("bix-10", "selected"), ("bix-54", "excluded"), ("bix-51", "excluded")]  # fmt: skip


def test_biomysterybench_screen_on_adapter_extra_fact(bmb_raw):
    # Arrange
    cfg = {"benchmark": "biomysterybench",
           "rules": [{"id": "solvable", "fact": "extra.human_solvable", "in": ["yes"]}]}  # fmt: skip
    # Act
    result = screen_eligibility(cfg, raw_dir=bmb_raw)
    # Assert
    assert sorted(result["selected"]) == ["hb002", "hb020", "recqgsfxqqodhjens"]


def test_fetch_stages_manifest_and_archives_from_an_empty_root(tmp_path):
    # Arrange
    cfg = {"benchmark": "biomysterybench",
           "rules": [{"id": "entry", "fact": "entry_kinds", "any_of": ["python", "notebook"]}]}  # fmt: skip
    # Act
    result = _staged(
        lambda: screen_eligibility(cfg, raw_dir=tmp_path / "raw", fetch=True)
    )
    # Assert
    assert result["counts"] == {"excluded": 5}


def test_cli_screen_writes_the_result_file(cb_raw, tmp_path):
    # Arrange
    from scitex_dataset._cli import main

    cfg = tmp_path / "criteria.json"
    cfg.write_text(json.dumps(CB_CONFIG))
    out = tmp_path / "eligibility.json"
    # Act
    res = CliRunner().invoke(main, ["ai-for-science", "corebench", "screen", "--config",
                                    str(cfg), "--raw-dir", str(cb_raw), "--out", str(out)])  # fmt: skip
    # Assert
    assert (res.exit_code, json.loads(out.read_text())["selected"]) == (
        0,
        ["capsule-6460826"],
    )
