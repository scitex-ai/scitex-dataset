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


@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_prepare_honors_selected_capsule_and_acquisition_options(module, tmp_path):
    calls = []
    with local_snapshot(calls):
        result = module.prepare(paths=paths_for(module, tmp_path), only='b', force=True, hf_token='synthetic-token', max_workers=1, revision='a' * 40)
    assert result['standardize']['for_solver']['n_materialized'] == 1
    assert result['standardize']['for_solver']['only'] == 'b'
    assert len(calls) == 1
    assert calls[0]['token'] == 'synthetic-token'
    assert calls[0]['max_workers'] == 1
    assert calls[0]['revision'] == 'a' * 40
    assert result['download']['requested_revision'] == 'a' * 40


@pytest.mark.parametrize('module', [bixbench, biomysterybench])
def test_direct_download_honors_requested_revision(module, tmp_path):
    calls = []
    with local_snapshot(calls):
        result = module.download(raw_dir=tmp_path / 'raw', revision='b' * 40)
    assert calls[0]['revision'] == 'b' * 40
    assert result['requested_revision'] == 'b' * 40


def test_biomystery_full_is_isolated_and_manifest_names_actual_variant(tmp_path):
    paths = paths_for(biomysterybench, tmp_path)
    calls = []
    with local_snapshot(calls):
        preview = biomysterybench.prepare(paths=paths)
        preview_bytes = (Path(preview['download']['raw_dir']) / 'acquisition-source.txt').read_bytes()
        preview_answers = Path(preview['standardize']['eval']['answers']).read_bytes()
        calls.clear()
        full = biomysterybench.prepare(paths=paths, download_full=True)
    assert [call['repo_id'] for call in calls] == [biomysterybench.HF_REPO_ID_FULL]
    assert full['download']['raw_dir'] != preview['download']['raw_dir']
    assert full['paths']['raw_dir'] == full['download']['raw_dir']
    assert (Path(preview['download']['raw_dir']) / 'acquisition-source.txt').read_bytes() == preview_bytes
    assert full['standardize']['eval']['answers'] != preview['standardize']['eval']['answers']
    assert full['standardize']['for_solver']['index'] != preview['standardize']['for_solver']['index']
    assert full['manifest'] != preview['manifest']
    assert Path(preview['standardize']['eval']['answers']).read_bytes() == preview_answers
    assert (Path(full['download']['raw_dir']) / 'acquisition-source.txt').read_text() == biomysterybench.HF_REPO_ID_FULL
    source_lines = [line.removeprefix('source_url: ') for line in Path(full['manifest']).read_text().splitlines() if line.startswith('source_url: ')]
    expected_url = f'https://huggingface.co/datasets/{biomysterybench.HF_REPO_ID_FULL}'
    assert source_lines in [[expected_url], [json.dumps(expected_url)]]
