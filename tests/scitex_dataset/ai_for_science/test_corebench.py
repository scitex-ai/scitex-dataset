#!/usr/bin/env python3
"""Tests for ai_for_science.corebench standardize + inventory + download.

PA-306 compliance: no ``unittest.mock``, no ``monkeypatch``. The network
download uses the module-level ``_corebench_download._http_download``
helper, which we replace with a hand-rolled stub via attribute
save/restore for the duration of each test. (The capsule fetch + oracle
bootstrap moved to ``_corebench_download``; ``corebench.download`` is a
re-export, so the seam lives on that module.)
"""

import json
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import _corebench_download, corebench

# ---------------------------------------------------------------------------
# Network seam — swap _corebench_download._http_download (no unittest.mock)
# ---------------------------------------------------------------------------


@contextmanager
def _swap_http_download(replacement):
    """Replace ``_corebench_download._http_download`` for the block."""
    saved = _corebench_download._http_download
    _corebench_download._http_download = replacement  # type: ignore[assignment]
    try:
        yield
    finally:
        _corebench_download._http_download = saved  # type: ignore[assignment]


@pytest.fixture
def three_reference_runs():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    tasks, answers = corebench._split_record(_reference_record([{'Score?': 12.34}, {'Score?': 12.35}, {'Score?': 12.33}]))
    return {'answers': answers, 'tasks': tasks}

@pytest.fixture
def reordered_reference_runs():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    first = _reference_record([{'A?': 1, 'B?': 4}, {'B?': 5, 'A?': 2}])
    reordered = _reference_record([{'A?': 2, 'B?': 5}, {'B?': 4, 'A?': 1}])
    tasks, answers = corebench._split_record(first)
    new_tasks, new_answers = corebench._split_record(reordered)
    old_samples = sorted(((row['task_id'], row['answer']['value']) for row in answers))
    new_samples = sorted(((row['task_id'], row['answer']['value']) for row in new_answers))
    return {'answers': answers, 'first': first, 'new_answers': new_answers, 'new_samples': new_samples, 'new_tasks': new_tasks, 'old_samples': old_samples, 'reordered': reordered, 'tasks': tasks}

@pytest.fixture
def extended_reference_runs():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    tasks, _ = corebench._split_record(_reference_record([{'A?': 10}]))
    extended_tasks, answers = corebench._split_record(_reference_record([{'A?': 10}, {'A?': 10}, {'A?': 11}]))
    return {'_': _, 'answers': answers, 'extended_tasks': extended_tasks, 'tasks': tasks}

@pytest.fixture
def missing_reference_runs():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    tasks, answers = corebench._split_record(_reference_record([{'A?': 1, 'B?': 9}, {'B?': 8}, {'A?': 2, 'B?': 7}]))
    a_refs = [row for row in answers if row['meta']['question'] == 'A?']
    b_refs = [row for row in answers if row['meta']['question'] == 'B?']
    return {'a_refs': a_refs, 'answers': answers, 'b_refs': b_refs, 'tasks': tasks}

@pytest.fixture
def exact_reference_keys():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    tasks, _ = corebench._split_record(_reference_record([{'A?': 1, ' A?': 2}]))
    changed_answers, _ = corebench._split_record(_reference_record([{'A?': 100, ' A?': None}]))
    return {'_': _, 'changed_answers': changed_answers, 'tasks': tasks}

@pytest.fixture
def invalid_utf8_reference():
    """Function-scoped original synthetic workflow; records facts without assertions."""
    _observed_exception = None
    try:
        corebench._split_record(_reference_record([{'\ud800': 123456789}]))
    except ValueError as caught_error:
        _observed_exception = caught_error
    return {'_observed_exception': _observed_exception}

@pytest.fixture
def materialized_question_ids(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    root = staged_raw_dir.parent
    corebench.standardize(raw_dir=staged_raw_dir, for_solver_dir=root / 'for_solver', eval_dir=root / 'eval')
    ids = _all_task_ids(root / 'for_solver')
    return {'ids': ids, 'root': root, 'staged_raw_dir': staged_raw_dir}

@pytest.fixture
def materialized_reference_question(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    (staged_raw_dir / 'dataset' / 'core_train.json').write_text(json.dumps([_reference_record([{'Score?': 12.34}, {'Score?': 12.35}, {'Score?': 12.33}])]))
    (staged_raw_dir / 'core_test.json').write_text('[]')
    root = staged_raw_dir.parent
    result = corebench.standardize(raw_dir=staged_raw_dir, for_solver_dir=root / 'for_solver', eval_dir=root / 'eval')
    tasks = _read_jsonl(root / 'for_solver' / 'capsule-001' / 'task.jsonl')
    answers = _read_jsonl(root / 'eval' / 'answers.jsonl')
    example = json.loads((root / 'for_solver' / 'capsule-001' / 'submission.example.json').read_text())
    return {'answers': answers, 'example': example, 'result': result, 'root': root, 'staged_raw_dir': staged_raw_dir, 'tasks': tasks}

@pytest.fixture
def duplicate_reference_questions(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    _observed_exception = None
    record = _reference_record([{'Score?': 12.34}])
    (staged_raw_dir / 'dataset' / 'core_train.json').write_text(json.dumps([record, record]))
    (staged_raw_dir / 'core_test.json').write_text('[]')
    root = staged_raw_dir.parent
    try:
        corebench.standardize(raw_dir=staged_raw_dir, for_solver_dir=root / 'for_solver', eval_dir=root / 'eval')
    except ValueError as caught_error:
        _observed_exception = caught_error
    return {'_observed_exception': _observed_exception, 'record': record, 'root': root, 'staged_raw_dir': staged_raw_dir}

@pytest.fixture
def question_reference_inventory(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    (staged_raw_dir / 'dataset' / 'core_train.json').write_text(json.dumps([_reference_record([{'Score?': 1}, {'Score?': 2}, {'Score?': 3}])]))
    (staged_raw_dir / 'core_test.json').write_text('[]')
    result = corebench.build_inventory(raw_dir=staged_raw_dir, for_solver_dir=staged_raw_dir.parent / 'for_solver')
    inventory = json.loads(Path(result['output']).read_text())
    return {'inventory': inventory, 'result': result, 'staged_raw_dir': staged_raw_dir}

@pytest.fixture
def refused_legacy_prepare(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    _observed_exception = None
    paths = _paths_for(staged_raw_dir)
    paths.for_solver_dir.mkdir()
    inventory = paths.for_solver_dir / 'inventory.json'
    old_inventory = b'{"historical": "original question units"}\n'
    inventory.write_bytes(old_inventory)
    try:
        corebench.prepare(paths=paths, skip_download=True, force=True)
    except ValueError as caught_error:
        _observed_exception = caught_error
    return {'_observed_exception': _observed_exception, 'inventory': inventory, 'old_inventory': old_inventory, 'paths': paths, 'staged_raw_dir': staged_raw_dir}

@pytest.fixture
def selected_capsule_prepare(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    paths = _paths_for(staged_raw_dir)
    result = corebench.prepare(paths=paths, capsule_ids=['capsule-2222222'], only='capsule-2222222')
    return {'paths': paths, 'result': result, 'staged_raw_dir': staged_raw_dir}

@pytest.fixture
def unknown_selector_prepare(staged_raw_dir):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    _observed_exception = None
    paths = _paths_for(staged_raw_dir)
    try:
        corebench.prepare(paths=paths, capsule_id='capsule-2222222')
    except TypeError as caught_error:
        _observed_exception = caught_error
    return {'_observed_exception': _observed_exception, 'paths': paths, 'staged_raw_dir': staged_raw_dir}

class _HttpRecorder:
    """Records fetch calls and writes deterministic bytes to each dest."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append((url, str(dest)))
        Path(dest).write_bytes(b"capsule-content")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_train_record():
    return {
        "capsule_id": "capsule-1111111",
        "language": "Python",
        "field": "biology",
        "task_prompt": "Read the README.",
        "capsule_title": "A Paper About Stuff",
        "capsule_doi": "10.1234/foo",
        "results": [
            {"What is the AUC?": 0.81},
            {"How many trials?": 12},
            {"What pH?": 7.4},
        ],
    }


def _make_test_record():
    return {
        "capsule_id": "capsule-2222222",
        "language": "R",
        "field": "ecology",
        "task_prompt": "Run the R script.",
        "capsule_title": "Another Paper",
        "capsule_doi": "10.5678/bar",
        "results": [
            {"What is the mean?": 4.2},
            {"What is the variance?": 0.5},
        ],
    }


def _write_capsule_targz(path: Path, cid: str) -> None:
    """Write a minimal real ``.tar.gz`` for capsule ``cid`` at ``path``.

    The per-capsule materializer EXTRACTS each capsule's archive, so the
    test fixtures must be genuine tarballs (not placeholder text).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / f"_scratch_{cid}"
    (scratch / "code").mkdir(parents=True, exist_ok=True)
    (scratch / "code" / "main.py").write_text(f"# {cid}\nprint('hi')\n")
    (scratch / "ReadMe").write_text(f"{cid} readme\n")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(scratch / "code" / "main.py", arcname="code/main.py")
        tf.add(scratch / "ReadMe", arcname="ReadMe")


@pytest.fixture
def staged_raw_dir(tmp_path):
    """Lay out raw/ with oracle JSONs + real capsule tarballs to extract."""
    base = tmp_path / "ai-for-science" / "corebench" / "raw"
    (base / "dataset").mkdir(parents=True)
    (base / "dataset" / "core_train.json").write_text(
        json.dumps([_make_train_record()])
    )
    (base / "core_test.json").write_text(json.dumps([_make_test_record()]))
    (base / "capsules").mkdir()
    _write_capsule_targz(
        base / "capsules" / "capsule-1111111.tar.gz", "capsule-1111111"
    )
    _write_capsule_targz(
        base / "capsules" / "capsule-2222222.tar.gz", "capsule-2222222"
    )
    return base


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _all_task_ids(for_solver_dir):
    """Collect task_ids across every per-capsule ``task.jsonl``."""
    ids = set()
    for task_file in for_solver_dir.glob("capsule-*/task.jsonl"):
        for row in _read_jsonl(task_file):
            ids.add(row["task_id"])
    return ids


def _reference_record(results):
    """Invented reference runs; never a real benchmark oracle."""
    return {
        "capsule_id": "capsule-1111111",
        "task_prompt": "Run the supplied analysis.",
        "results": results,
    }


class TestQuestionReferenceIdentity:
    def test_three_reference_runs_create_one_question(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert len(tasks) == 1

    def test_reference_rows_share_one_question_id(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert len({answer['task_id'] for answer in answers}) == 1

    def test_reference_rows_preserve_every_sample_value(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert [answer['answer']['value'] for answer in answers] == [12.34, 12.35, 12.33]

    def test_reference_rows_preserve_original_run_indexes(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert [answer['meta']['reference_run_index'] for answer in answers] == [0, 1, 2]

    def test_private_reference_ids_match_public_question(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert all((answer['task_id'] == tasks[0]['task_id'] for answer in answers))

    def test_reference_rows_declare_question_sample_schema(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert all((answer['meta']['schema'] == 'corebench-question-references-v1' for answer in answers))

    def test_reference_metadata_omits_synthetic_difficulty(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert all(('difficulty' not in answer['meta'] for answer in answers))

    def test_public_question_has_only_uniform_fields(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert all((set(task) == {'task_id', 'benchmark', 'prompt', 'data'} for task in tasks))

    def test_public_question_omits_private_reference_value(self, three_reference_runs):
        # Arrange
        observed = three_reference_runs
        # Act
        answers = observed['answers']
        tasks = observed['tasks']
        # Assert
        assert '12.34' not in json.dumps(tasks)

    def test_reordered_reference_runs_retain_two_questions(self, reordered_reference_runs):
        # Arrange
        observed = reordered_reference_runs
        # Act
        answers = observed['answers']
        first = observed['first']
        new_answers = observed['new_answers']
        new_samples = observed['new_samples']
        new_tasks = observed['new_tasks']
        old_samples = observed['old_samples']
        reordered = observed['reordered']
        tasks = observed['tasks']
        # Assert
        assert len(tasks) == 2

    def test_reordered_reference_runs_keep_public_assignment(self, reordered_reference_runs):
        # Arrange
        observed = reordered_reference_runs
        # Act
        answers = observed['answers']
        first = observed['first']
        new_answers = observed['new_answers']
        new_samples = observed['new_samples']
        new_tasks = observed['new_tasks']
        old_samples = observed['old_samples']
        reordered = observed['reordered']
        tasks = observed['tasks']
        # Assert
        assert tasks == new_tasks

    def test_reordered_reference_runs_keep_sample_multiset(self, reordered_reference_runs):
        # Arrange
        observed = reordered_reference_runs
        # Act
        answers = observed['answers']
        first = observed['first']
        new_answers = observed['new_answers']
        new_samples = observed['new_samples']
        new_tasks = observed['new_tasks']
        old_samples = observed['old_samples']
        reordered = observed['reordered']
        tasks = observed['tasks']
        # Assert
        assert old_samples == new_samples

    def test_adding_reference_keeps_public_question_assignment(self, extended_reference_runs):
        # Arrange
        observed = extended_reference_runs
        # Act
        _ = observed['_']
        answers = observed['answers']
        extended_tasks = observed['extended_tasks']
        tasks = observed['tasks']
        # Assert
        assert tasks == extended_tasks

    def test_adding_reference_keeps_duplicate_sample_values(self, extended_reference_runs):
        # Arrange
        observed = extended_reference_runs
        # Act
        _ = observed['_']
        answers = observed['answers']
        extended_tasks = observed['extended_tasks']
        tasks = observed['tasks']
        # Assert
        assert [row['answer']['value'] for row in answers] == [10, 10, 11]

    def test_adding_reference_updates_sample_count_metadata(self, extended_reference_runs):
        # Arrange
        observed = extended_reference_runs
        # Act
        _ = observed['_']
        answers = observed['answers']
        extended_tasks = observed['extended_tasks']
        tasks = observed['tasks']
        # Assert
        assert all((row['meta']['reference_count'] == 3 for row in answers))

    def test_missing_reference_run_retains_both_questions(self, missing_reference_runs):
        # Arrange
        observed = missing_reference_runs
        # Act
        a_refs = observed['a_refs']
        answers = observed['answers']
        b_refs = observed['b_refs']
        tasks = observed['tasks']
        # Assert
        assert len(tasks) == 2

    def test_missing_reference_run_preserves_available_values(self, missing_reference_runs):
        # Arrange
        observed = missing_reference_runs
        # Act
        a_refs = observed['a_refs']
        answers = observed['answers']
        b_refs = observed['b_refs']
        tasks = observed['tasks']
        # Assert
        assert [row['answer']['value'] for row in a_refs] == [1, 2]

    def test_missing_reference_run_reports_missing_index(self, missing_reference_runs):
        # Arrange
        observed = missing_reference_runs
        # Act
        a_refs = observed['a_refs']
        answers = observed['answers']
        b_refs = observed['b_refs']
        tasks = observed['tasks']
        # Assert
        assert all((row['meta']['missing_reference_run_indexes'] == [1] for row in a_refs))

    def test_missing_reference_run_retains_total_run_count(self, missing_reference_runs):
        # Arrange
        observed = missing_reference_runs
        # Act
        a_refs = observed['a_refs']
        answers = observed['answers']
        b_refs = observed['b_refs']
        tasks = observed['tasks']
        # Assert
        assert all((row['meta']['reference_runs_total'] == 3 for row in answers))

    def test_complete_reference_question_has_no_missing_indexes(self, missing_reference_runs):
        # Arrange
        observed = missing_reference_runs
        # Act
        a_refs = observed['a_refs']
        answers = observed['answers']
        b_refs = observed['b_refs']
        tasks = observed['tasks']
        # Assert
        assert all((row['meta']['missing_reference_run_indexes'] == [] for row in b_refs))

    @pytest.mark.parametrize("results", [None, [], [None], [1], [{"": 1}], [{}]])
    def test_invalid_reference_source_refuses_instead_of_dropping(self, results):
        # Arrange
        record = _reference_record(results)
        # Act
        error_context = pytest.raises(ValueError, match="corebench:")
        # Assert
        with error_context:
            corebench._split_record(record)

    def test_distinct_exact_question_keys_have_distinct_ids(self, exact_reference_keys):
        # Arrange
        observed = exact_reference_keys
        # Act
        _ = observed['_']
        changed_answers = observed['changed_answers']
        tasks = observed['tasks']
        # Assert
        assert len({task['task_id'] for task in tasks}) == 2

    def test_question_ids_do_not_depend_on_reference_values(self, exact_reference_keys):
        # Arrange
        observed = exact_reference_keys
        # Act
        _ = observed['_']
        changed_answers = observed['changed_answers']
        tasks = observed['tasks']
        # Assert
        assert tasks == changed_answers

    def test_invalid_utf8_question_key_raises_attributable_error(self):
        # Arrange
        record = None
        # Act
        error_context = pytest.raises(ValueError, match='question key must be valid UTF-8')
        # Assert
        with error_context:
            corebench._split_record(_reference_record([{'\ud800': 123456789}]))

    def test_invalid_utf8_question_error_omits_private_value(self, invalid_utf8_reference):
        # Arrange
        observed = invalid_utf8_reference
        # Act
        _observed_exception = observed['_observed_exception']
        # Assert
        assert '123456789' not in str(_observed_exception)


# ---------------------------------------------------------------------------
# standardize — for_solver tasks (leak-safe uniform schema)
# ---------------------------------------------------------------------------


class TestStandardizeForSolver:
    def test_standardize_writes_index_mapper(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "index.jsonl").is_file()

    def test_index_maps_native_id_to_friendly_id(self, staged_raw_dir):
        # Arrange — capsule-1111111 sorts first, so it gets capsule-001.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        index = _read_jsonl(root / "for_solver" / "index.jsonl")
        row = next(r for r in index if r["native_id"] == "capsule-1111111")
        # Assert
        assert row["friendly_id"] == "capsule-001"

    def test_standardize_writes_per_capsule_task_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "capsule-001" / "task.jsonl").is_file()

    def test_tasks_have_exactly_uniform_keys(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Assert
        assert all(set(t) == {"task_id", "benchmark", "prompt", "data"} for t in tasks)

    def test_tasks_carry_no_answer_value(self, staged_raw_dir):
        # Arrange — the oracle answer 0.81 must not appear in any task row.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        raw_text = (root / "for_solver" / "capsule-001" / "task.jsonl").read_text()
        # Assert
        assert "0.81" not in raw_text

    def test_task_id_uses_stable_question_identity(self, materialized_question_ids):
        # Arrange
        observed = materialized_question_ids
        # Act
        ids = observed['ids']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert corebench._question_task_id('capsule-1111111', 'What is the AUC?') in ids

    def test_materialized_task_ids_use_question_grammar(self, materialized_question_ids):
        # Arrange
        observed = materialized_question_ids
        # Act
        ids = observed['ids']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert all(('__question_' in task_id for task_id in ids))

    def test_materialized_task_ids_omit_positional_difficulty(self, materialized_question_ids):
        # Arrange
        observed = materialized_question_ids
        # Act
        ids = observed['ids']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert all(('__hard__' not in task_id for task_id in ids))

    def test_task_data_points_at_extracted_input(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Act
        data_values = {t["data"] for t in tasks}
        # Assert
        assert data_values == {"./input"}

    def test_capsule_archive_extracted_into_input(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Act
        extracted = root / "for_solver" / "capsule-001" / "input" / "code" / "main.py"
        # Assert
        assert extracted.is_file()

    def test_task_prompt_appends_question_text(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        tasks = _read_jsonl(root / "for_solver" / "capsule-001" / "task.jsonl")
        # Act
        task_id = corebench._question_task_id("capsule-1111111", "What is the AUC?")
        task = next(t for t in tasks if t["task_id"] == task_id)
        # Assert
        assert task["prompt"].endswith("Question: What is the AUC?")

    def test_standardize_writes_submission_schema_per_capsule(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (
            root / "for_solver" / "capsule-001" / "submission.schema.json"
        ).is_file()

    def test_standardize_writes_capsule_readme(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "for_solver" / "capsule-001" / "README.md").is_file()

    def test_only_native_id_materializes_single_capsule(self, staged_raw_dir):
        # Arrange — request only the test-split capsule by native id.
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
            only="capsule-2222222",
        )
        # Assert — capsule-2222222 sorts second → capsule-002 only.
        assert (root / "for_solver" / "capsule-002").is_dir() and not (
            root / "for_solver" / "capsule-001"
        ).exists()

    def test_standardize_does_not_expose_oracle_dataset_dir(self, staged_raw_dir):
        # Arrange — the answer-bearing ``dataset`` dir must never appear.
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert not (root / "for_solver" / "dataset").exists()

    def test_standardize_raises_when_oracle_train_missing(self, tmp_path):
        # Arrange
        bare = tmp_path / "empty-raw"
        bare.mkdir()
        # Act
        # Assert
        with pytest.raises(FileNotFoundError):
            corebench.standardize(
                raw_dir=bare,
                for_solver_dir=tmp_path / "fs",
                eval_dir=tmp_path / "ev",
            )


# ---------------------------------------------------------------------------
# standardize — eval answers (operator view)
# ---------------------------------------------------------------------------


class TestStandardizeEval:
    def test_standardized_question_and_example_counts_match(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert result['n_tasks'] == len(tasks) == len(example) == 1

    def test_standardized_reference_count_matches_oracle_rows(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert result['n_reference_samples'] == len(answers) == 3

    def test_standardized_oracle_preserves_every_sample_value(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert [row['answer']['value'] for row in answers] == [12.34, 12.35, 12.33]

    def test_standardized_oracle_ids_match_public_question(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert {row['task_id'] for row in answers} == {tasks[0]['task_id']}

    def test_standardized_public_question_and_example_omit_values(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert '12.34' not in json.dumps(tasks + example)

    def test_standardized_public_question_omits_private_source_identity(self, materialized_reference_question):
        # Arrange
        observed = materialized_reference_question
        # Act
        answers = observed['answers']
        example = observed['example']
        result = observed['result']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        tasks = observed['tasks']
        # Assert
        assert 'source_identity' not in tasks[0]

    def test_duplicate_question_records_raise_identity_error(self, staged_raw_dir):
        # Arrange
        record = _reference_record([{'Score?': 12.34}])
        (staged_raw_dir / 'dataset' / 'core_train.json').write_text(json.dumps([record, record]))
        (staged_raw_dir / 'core_test.json').write_text('[]')
        root = staged_raw_dir.parent
        # Act
        error_context = pytest.raises(ValueError, match='duplicate assigned question identity')
        # Assert
        with error_context:
            corebench.standardize(raw_dir=staged_raw_dir, for_solver_dir=root / 'for_solver', eval_dir=root / 'eval')

    def test_duplicate_questions_leave_solver_directory_absent(self, duplicate_reference_questions):
        # Arrange
        observed = duplicate_reference_questions
        # Act
        _observed_exception = observed['_observed_exception']
        record = observed['record']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not (root / 'for_solver').exists()

    def test_duplicate_questions_leave_evaluator_directory_absent(self, duplicate_reference_questions):
        # Arrange
        observed = duplicate_reference_questions
        # Act
        _observed_exception = observed['_observed_exception']
        record = observed['record']
        root = observed['root']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not (root / 'eval').exists()

    def test_standardize_writes_answers_jsonl(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "answers.jsonl").is_file()

    def test_answer_task_ids_match_task_ids(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        task_ids = _all_task_ids(root / "for_solver")
        # Act
        answer_ids = {
            a["task_id"] for a in _read_jsonl(root / "eval" / "answers.jsonl")
        }
        # Assert
        assert answer_ids == task_ids

    def test_answers_carry_value_payload(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        # Act
        task_id = corebench._question_task_id("capsule-1111111", "What is the AUC?")
        answer = next(a for a in answers if a["task_id"] == task_id)
        # Assert
        assert answer["answer"] == {"value": 0.81}

    def test_standardize_writes_evaluate_py(self, staged_raw_dir):
        # Arrange
        root = staged_raw_dir.parent
        # Act
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        # Assert
        assert (root / "eval" / "evaluate.py").is_file()


class TestEvaluatePyRoundTrip:
    def test_correct_submission_scores_one(self, staged_raw_dir):
        # Arrange — build a submission that matches every oracle value.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        sub = [
            {"task_id": a["task_id"], "answer": a["answer"]["value"]} for a in answers
        ]
        (root / "good.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(root / "eval" / "evaluate.py"),
                "--submission",
                str(root / "good.json"),
                "--answers",
                str(root / "eval" / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["score"] == 1.0

    def test_wrong_submission_scores_below_one(self, staged_raw_dir):
        # Arrange — every answer deliberately wrong.
        root = staged_raw_dir.parent
        corebench.standardize(
            raw_dir=staged_raw_dir,
            for_solver_dir=root / "for_solver",
            eval_dir=root / "eval",
        )
        answers = _read_jsonl(root / "eval" / "answers.jsonl")
        sub = [{"task_id": a["task_id"], "answer": -99999} for a in answers]
        (root / "bad.json").write_text(json.dumps(sub))
        # Act
        out = subprocess.run(
            [
                sys.executable,
                str(root / "eval" / "evaluate.py"),
                "--submission",
                str(root / "bad.json"),
                "--answers",
                str(root / "eval" / "answers.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        # Assert
        assert json.loads(out.stdout)["score"] < 1.0


# ---------------------------------------------------------------------------
# build_inventory — writes to for_solver
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def test_build_inventory_writes_inventory_json_in_for_solver(self, staged_raw_dir):
        # Arrange
        for_solver_dir = staged_raw_dir.parent / "for_solver"
        # Act
        corebench.build_inventory(raw_dir=staged_raw_dir, for_solver_dir=for_solver_dir)
        # Assert
        assert (for_solver_dir / "inventory.json").is_file()

    def test_build_inventory_summary_counts_one_train_capsule(self, staged_raw_dir):
        # Arrange
        for_solver_dir = staged_raw_dir.parent / "for_solver"
        # Act
        result = corebench.build_inventory(
            raw_dir=staged_raw_dir, for_solver_dir=for_solver_dir
        )
        # Assert
        assert result["summary"]["n_capsules_train"] == 1

    def test_inventory_counts_unique_questions_once(self, question_reference_inventory):
        # Arrange
        observed = question_reference_inventory
        # Act
        inventory = observed['inventory']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert result['summary']['n_tasks_total'] == 1

    def test_inventory_counts_every_reference_sample(self, question_reference_inventory):
        # Arrange
        observed = question_reference_inventory
        # Act
        inventory = observed['inventory']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert result['summary']['n_reference_samples'] == 3

    def test_inventory_summary_has_no_synthetic_difficulty(self, question_reference_inventory):
        # Arrange
        observed = question_reference_inventory
        # Act
        inventory = observed['inventory']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert result['summary']['by_difficulty'] == {}

    def test_inventory_question_retains_reference_sample_count(self, question_reference_inventory):
        # Arrange
        observed = question_reference_inventory
        # Act
        inventory = observed['inventory']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert inventory['tasks'][0]['reference_count'] == 3

    def test_inventory_question_has_no_positional_difficulty(self, question_reference_inventory):
        # Arrange
        observed = question_reference_inventory
        # Act
        inventory = observed['inventory']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert inventory['tasks'][0]['difficulty'] is None


# ---------------------------------------------------------------------------
# download — checksum-verified skip
# ---------------------------------------------------------------------------


class TestDownloadChecksumSkip:
    def test_first_run_fetches_each_capsule(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_fetched"] == 2

    def test_second_run_default_skips_by_existence(self, tmp_path):
        # Arrange — default policy skips present files with NO hashing.
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_have"] == 2

    def test_second_run_skips_verified_when_opt_in(self, tmp_path):
        # Arrange — verify_integrity re-checks sha256 against the ledger.
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], verify_integrity=True
            )
        # Assert
        assert result["n_skipped_verified"] == 2

    def test_second_run_fetches_nothing(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert result["n_fetched"] == 0

    def test_tampered_capsule_is_refetched_under_verify(self, tmp_path):
        # Arrange — corrupt a verified file so its sha drifts (only the
        # opt-in integrity pass detects it; default skip would not).
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        (raw_dir / "capsules" / "111.tar.gz").write_bytes(b"TAMPERED")
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], verify_integrity=True
            )
        # Assert
        assert result["n_remismatch"] == 1

    def test_force_refetches_existing(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Act
        with _swap_http_download(rec):
            result = corebench.download(
                raw_dir=raw_dir, capsule_ids=["111", "222"], force=True
            )
        # Assert
        assert result["n_fetched"] == 2

    def test_download_writes_checksums_ledger(self, tmp_path):
        # Arrange
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111"])
        # Assert
        assert (raw_dir / ".checksums.json").is_file()

    def test_explicit_capsule_ids_never_touch_oracle_urls(self, tmp_path):
        # Arrange — an explicit id list must stay oracle-free (no upstream
        # raw.githubusercontent.com fetch, no gpg).
        rec = _HttpRecorder()
        raw_dir = tmp_path / "raw"
        # Act
        with _swap_http_download(rec):
            corebench.download(raw_dir=raw_dir, capsule_ids=["111", "222"])
        # Assert
        assert all("githubusercontent" not in url for url, _ in rec.calls)


# ---------------------------------------------------------------------------
# prepare orchestrator
# ---------------------------------------------------------------------------


def _paths_for(staged_raw_dir):
    from scitex_dataset.ai_for_science import _base

    root = staged_raw_dir.parent
    return _base.BenchmarkPaths(
        benchmark=corebench.BENCHMARK,
        root=root,
        raw_dir=staged_raw_dir,
        for_solver_dir=root / "for_solver",
        eval_dir=root / "eval",
        manifest_dir=root / ".scitex" / "dataset",
    )


class TestPrepare:
    def test_legacy_prepare_force_raises_cache_refusal(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        paths.for_solver_dir.mkdir()
        inventory = paths.for_solver_dir / 'inventory.json'
        old_inventory = b'{"historical": "original question units"}\n'
        inventory.write_bytes(old_inventory)
        # Act
        error_context = pytest.raises(ValueError, match='legacy cache')
        # Assert
        with error_context:
            corebench.prepare(paths=paths, skip_download=True, force=True)

    def test_legacy_prepare_preserves_original_inventory_bytes(self, refused_legacy_prepare):
        # Arrange
        observed = refused_legacy_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        inventory = observed['inventory']
        old_inventory = observed['old_inventory']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert inventory.read_bytes() == old_inventory

    def test_legacy_prepare_preserves_original_directory_contents(self, refused_legacy_prepare):
        # Arrange
        observed = refused_legacy_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        inventory = observed['inventory']
        old_inventory = observed['old_inventory']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert list(paths.for_solver_dir.iterdir()) == [inventory]

    def test_legacy_prepare_leaves_evaluator_directory_absent(self, refused_legacy_prepare):
        # Arrange
        observed = refused_legacy_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        inventory = observed['inventory']
        old_inventory = observed['old_inventory']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not paths.eval_dir.exists()

    def test_legacy_prepare_leaves_manifest_directory_absent(self, refused_legacy_prepare):
        # Arrange
        observed = refused_legacy_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        inventory = observed['inventory']
        old_inventory = observed['old_inventory']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not paths.manifest_dir.exists()

    def test_prepare_skip_download_emits_manifest_yaml(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert Path(result["manifest"]).is_file()

    def test_prepare_skip_download_has_standardize_key(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        # Act
        result = corebench.prepare(paths=paths, skip_download=True)
        # Assert
        assert "standardize" in result

    def test_prepare_with_download_emits_expected_keys(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        rec = _HttpRecorder()
        # Act
        with _swap_http_download(rec):
            result = corebench.prepare(paths=paths, skip_download=False)
        # Assert
        assert {"download", "inventory", "standardize", "manifest"} <= set(result)

    def test_prepare_acquisition_selector_counts_one_retained_capsule(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert result['download']['n_have'] == 1

    def test_prepare_acquisition_selector_fetches_no_retained_capsule(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert result['download']['n_fetched'] == 0

    def test_prepare_acquisition_selector_omits_unrequested_oracle(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert 'oracle' not in result['download']

    def test_prepare_materialization_selector_creates_selected_capsule(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert (paths.for_solver_dir / 'capsule-002').is_dir()

    def test_prepare_materialization_selector_omits_other_capsule(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not (paths.for_solver_dir / 'capsule-001').exists()

    def test_prepare_materialization_selector_preserves_complete_mapper(self, selected_capsule_prepare):
        # Arrange
        observed = selected_capsule_prepare
        # Act
        paths = observed['paths']
        result = observed['result']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert len(_read_jsonl(paths.for_solver_dir / 'index.jsonl')) == 2

    def test_unknown_prepare_selector_raises_keyword_error(self, staged_raw_dir):
        # Arrange
        paths = _paths_for(staged_raw_dir)
        # Act
        error_context = pytest.raises(TypeError, match='capsule_id')
        # Assert
        with error_context:
            corebench.prepare(paths=paths, capsule_id='capsule-2222222')

    def test_unknown_prepare_selector_leaves_solver_directory_absent(self, unknown_selector_prepare):
        # Arrange
        observed = unknown_selector_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not paths.for_solver_dir.exists()

    def test_unknown_prepare_selector_leaves_checksum_file_absent(self, unknown_selector_prepare):
        # Arrange
        observed = unknown_selector_prepare
        # Act
        _observed_exception = observed['_observed_exception']
        paths = observed['paths']
        staged_raw_dir = observed['staged_raw_dir']
        # Assert
        assert not (paths.raw_dir / '.checksums.json').exists()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
