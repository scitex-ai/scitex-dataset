"""Real synthetic preparation with a local HF collaborator; no upstream data."""

import csv
import json
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from scitex_dataset.ai_for_science import biomysterybench, bixbench
from scitex_dataset.ai_for_science._base import BenchmarkPaths


@contextmanager
def local_snapshot(calls):
    def acquire(**kwargs):
        calls.append(kwargs)
        raw = Path(kwargs['local_dir'])
        raw.mkdir(parents=True, exist_ok=True)
        if kwargs['repo_id'] == bixbench.HF_REPO_ID:
            records = [{'question_id': f'{key}-q1', 'question': 'Report the synthetic result', 'data_folder': f'{key}.zip', 'answer': 'synthetic-eval-only'} for key in ['a', 'b']]
            (raw / bixbench.ORACLE_MANIFEST_NAME).write_text(''.join(json.dumps(row) + '\n' for row in records))
            archives = [raw / f'{key}.zip' for key in ['a', 'b']]
        else:
            with (raw / 'problems.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['id', 'question', 'answer_rubric'])
                writer.writeheader()
                writer.writerows({'id': key, 'question': 'Explain the synthetic observation', 'answer_rubric': 'evaluator-only-rubric'} for key in ['a', 'b'])
            archives = [raw / 'data' / f'{key}.zip' for key in ['a', 'b']]
        for archive in archives:
            archive.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, 'w') as file:
                file.writestr('code/analyze.py', 'print("synthetic input")\n')
                file.writestr('data/input.csv', 'input\n1\n')
        (raw / 'acquisition-source.txt').write_text(kwargs['repo_id'])
        return str(raw)

    module = types.ModuleType('huggingface_hub')
    module.snapshot_download = acquire
    prior = sys.modules.get('huggingface_hub')
    sys.modules['huggingface_hub'] = module
    try:
        yield
    finally:
        if prior is None:
            del sys.modules['huggingface_hub']
        else:
            sys.modules['huggingface_hub'] = prior


def paths_for(module, root):
    return BenchmarkPaths(benchmark=module.BENCHMARK, root=root, raw_dir=root / 'raw', for_solver_dir=root / 'for_solver', eval_dir=root / 'eval', manifest_dir=root / 'manifests')


@pytest.fixture
def selected_hf_preparation(module, tmp_path):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    calls = []
    with local_snapshot(calls):
        result = module.prepare(paths=paths_for(module, tmp_path), only='b', force=True, hf_token='synthetic-token', max_workers=1, revision='a' * 40)
    return {'calls': calls, 'module': module, 'result': result, 'tmp_path': tmp_path}

@pytest.fixture
def direct_hf_acquisition(module, tmp_path):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    calls = []
    with local_snapshot(calls):
        result = module.download(raw_dir=tmp_path / 'raw', revision='b' * 40)
    return {'calls': calls, 'module': module, 'result': result, 'tmp_path': tmp_path}

@pytest.fixture
def preview_then_full_preparation(tmp_path):
    """Function-scoped original synthetic workflow; records facts without assertions."""
    paths = paths_for(biomysterybench, tmp_path)
    calls = []
    with local_snapshot(calls):
        preview = biomysterybench.prepare(paths=paths)
        preview_bytes = (Path(preview['download']['raw_dir']) / 'acquisition-source.txt').read_bytes()
        preview_answers = Path(preview['standardize']['eval']['answers']).read_bytes()
        calls.clear()
        full = biomysterybench.prepare(paths=paths, download_full=True)
    source_lines = [line.removeprefix('source_url: ') for line in Path(full['manifest']).read_text().splitlines() if line.startswith('source_url: ')]
    expected_url = f'https://huggingface.co/datasets/{biomysterybench.HF_REPO_ID_FULL}'
    return {'calls': calls, 'expected_url': expected_url, 'full': full, 'paths': paths, 'preview': preview, 'preview_answers': preview_answers, 'preview_bytes': preview_bytes, 'source_lines': source_lines, 'tmp_path': tmp_path}

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_materializes_one_capsule(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert result['standardize']['for_solver']['n_materialized'] == 1

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_reports_requested_native_id(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert result['standardize']['for_solver']['only'] == 'b'

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_calls_acquisition_collaborator_once(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert len(calls) == 1

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_forwards_explicit_hf_token(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert calls[0]['token'] == 'synthetic-token'

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_forwards_download_worker_count(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert calls[0]['max_workers'] == 1

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_forwards_requested_revision(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert calls[0]['revision'] == 'a' * 40

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_selected_prepare_reports_requested_revision(module, selected_hf_preparation):
    # Arrange
    observed = selected_hf_preparation
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert result['download']['requested_revision'] == 'a' * 40


@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_direct_download_forwards_requested_revision(module, direct_hf_acquisition):
    # Arrange
    observed = direct_hf_acquisition
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert calls[0]['revision'] == 'b' * 40

@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_direct_download_reports_requested_revision(module, direct_hf_acquisition):
    # Arrange
    observed = direct_hf_acquisition
    # Act
    calls = observed['calls']
    module = observed['module']
    result = observed['result']
    tmp_path = observed['tmp_path']
    # Assert
    assert result['requested_revision'] == 'b' * 40


def test_full_prepare_acquires_only_full_repository(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert [call['repo_id'] for call in calls] == [biomysterybench.HF_REPO_ID_FULL]

def test_full_prepare_separates_raw_paths_from_preview(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert full['download']['raw_dir'] != preview['download']['raw_dir']

def test_full_prepare_reports_actual_acquisition_path(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert full['paths']['raw_dir'] == full['download']['raw_dir']

def test_full_prepare_preserves_preview_acquisition_bytes(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert (Path(preview['download']['raw_dir']) / 'acquisition-source.txt').read_bytes() == preview_bytes

def test_full_prepare_separates_evaluator_paths_from_preview(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert full['standardize']['eval']['answers'] != preview['standardize']['eval']['answers']

def test_full_prepare_separates_solver_mapper_from_preview(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert full['standardize']['for_solver']['index'] != preview['standardize']['for_solver']['index']

def test_full_prepare_separates_manifest_path_from_preview(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert full['manifest'] != preview['manifest']

def test_full_prepare_preserves_preview_oracle_bytes(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert Path(preview['standardize']['eval']['answers']).read_bytes() == preview_answers

def test_full_prepare_records_actual_repository_marker(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert (Path(full['download']['raw_dir']) / 'acquisition-source.txt').read_text() == biomysterybench.HF_REPO_ID_FULL

def test_full_prepare_manifest_names_actual_source_repository(preview_then_full_preparation):
    # Arrange
    observed = preview_then_full_preparation
    # Act
    calls = observed['calls']
    expected_url = observed['expected_url']
    full = observed['full']
    paths = observed['paths']
    preview = observed['preview']
    preview_answers = observed['preview_answers']
    preview_bytes = observed['preview_bytes']
    source_lines = observed['source_lines']
    tmp_path = observed['tmp_path']
    # Assert
    assert source_lines in [[expected_url], [json.dumps(expected_url)]]
