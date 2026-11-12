"""Real child controls for evaluator timeout evidence and the unchanged budget."""

import json
import subprocess
import sys

import pytest

from scitex_dataset.ai_for_science import _standardize
from tests._evaluator_process import run_evaluator


def _scorer_files(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text(_standardize.render_evaluate_py("numeric"))
    submitted = tmp_path / "submission.json"
    submitted.write_text(json.dumps([{"task_id": "owned/q1", "answer": 5}]))
    oracle = tmp_path / "answers.jsonl"
    oracle.write_text(json.dumps({"task_id": "owned/q1", "answer": {"value": 5}}) + "\n")
    return [sys.executable, str(script), "--submission", str(submitted), "--answers", str(oracle)]


def test_generated_scoring_keeps_its_result_without_import_trace(tmp_path):
    # Arrange
    command = _scorer_files(tmp_path)
    # Act
    result = run_evaluator(command, trace=False)
    result.check_returncode()
    # Assert
    assert json.loads(result.stdout)["score"] == 1.0


def test_opt_in_trace_observes_real_interpreter_imports(tmp_path):
    # Arrange
    command = _scorer_files(tmp_path)
    # Act
    result = run_evaluator(command, trace=True)
    result.check_returncode()
    # Assert
    assert "import time:" in result.stderr


def test_timeout_preserves_the_three_second_exception_contract(tmp_path):
    # Arrange
    script = tmp_path / "blocked.py"
    script.write_text("import time\ntime.sleep(30)\n")
    # Act
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_evaluator([sys.executable, str(script)], trace=False)
    # Assert
    assert raised.value.timeout == 3


def test_timeout_evidence_excludes_arbitrary_child_stderr(tmp_path, capsys):
    # Arrange
    script = tmp_path / "blocked.py"
    script.write_text("import sys,time\nsys.stderr.write('import time: 1 | 2 | OWNED_PRIVATE_SENTINEL\\n')\nsys.stderr.flush()\ntime.sleep(30)\n")
    # Act
    with pytest.raises(subprocess.TimeoutExpired):
        run_evaluator([sys.executable, str(script)], trace=True)
    stderr = capsys.readouterr().err
    evidence = json.loads(stderr.split("EVALUATOR_TIMEOUT ", 1)[1])
    # Assert
    assert (evidence["timeout_seconds"], bool(evidence["last_import_timings"]),
            "OWNED_PRIVATE_SENTINEL" in stderr) == (3, True, False)
