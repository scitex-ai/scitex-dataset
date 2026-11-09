"""Exercise real workflow shell control flow with inert command fixtures.

These fixtures verify command selection and failure propagation. They do not
qualify dependency resolution, scientific tests, publication or branch reviews.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/pypi-publish-and-github-release-on-tag.yml"
)
PROTOCOL = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

executable, arguments = Path(sys.argv[0]), sys.argv[1:]
name = "owned-python" if executable.parent.parent.name == ".venv" else executable.name
with open(os.environ["COMMAND_LOG"], "a") as stream:
    stream.write(json.dumps([name, arguments]) + "\n")
if name in {"python", "pip", "uv"}:
    if ".[all,dev]" in arguments and os.environ["REFUSE_PROFILE"] == "1":
        sys.exit(17)
elif name == "git" and arguments[:2] == ["rev-list", "--count"]:
    print(os.environ["AHEAD"])
elif name == "gh":
    if arguments[:2] == ["pr", "list"]:
        print(os.environ["EXISTING_PR"])
    elif arguments[:2] == ["pr", "create"]:
        print("https://github.com/example/owned-fixture/pull/42")
"""


def _step(job, prefix):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return next(
        step["run"]
        for step in workflow["jobs"][job]["steps"]
        if step.get("name", "").startswith(prefix)
    )


def _run(tmp_path, script, **values):
    directory = tmp_path / "bin"
    directory.mkdir()
    command = directory / "command-fixture"
    command.write_text(PROTOCOL)
    command.chmod(0o700)
    for name in ["python", "pip", "git", "gh", "uv", "pytest"]:
        (directory / name).symlink_to(command.name)
    owned_bin = tmp_path / ".venv" / "bin"
    owned_bin.mkdir(parents=True)
    (owned_bin / "python").symlink_to(command)
    log = tmp_path / "commands.jsonl"
    environment = dict(os.environ)
    environment.update(
        PATH=str(directory) + os.pathsep + os.defpath,
        COMMAND_LOG=str(log),
        REFUSE_PROFILE="0",
        AHEAD="1",
        EXISTING_PR="21",
        RELEASE_TEST_PYTHON="3.12",
        GITHUB_ENV=str(tmp_path / "github.env"),
        POSTGRES_SERVICE_PORT="32678",
    )
    environment.update(values)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls


@pytest.mark.parametrize("refuse,expected", [("1", 17), ("0", 0)])
def test_install_uses_one_profile_and_preserves_failure(tmp_path, refuse, expected):
    # Arrange
    script = _step("test", "Install")
    # Act
    result, calls = _run(tmp_path, script, REFUSE_PROFILE=refuse)
    profiles = [(name, arguments) for name, arguments in calls if "-e" in arguments]
    # Assert
    assert (result.returncode, profiles) == (
        expected,
        [
            (
                "uv",
                ["pip", "install", "--python", ".venv/bin/python", "-e", ".[all,dev]"],
            )
        ],
    )


@pytest.mark.parametrize("job", ["test", "build"])
def test_installer_never_invokes_the_runner_interpreter(tmp_path, job):
    # Arrange
    script = _step(job, "Install")
    # Act
    result, calls = _run(tmp_path, script)
    # Assert
    global_calls = [call for call in calls if call[0] in {"python", "pip"}]
    assert (result.returncode, global_calls) == (0, [])


@pytest.mark.parametrize("job", ["test", "build"])
def test_installer_targets_exactly_one_owned_interpreter(tmp_path, job):
    # Arrange
    script = _step(job, "Install")
    # Act
    result, calls = _run(tmp_path, script)
    installs = [
        args for name, args in calls if name == "uv" and args[:2] == ["pip", "install"]
    ]
    # Assert
    assert (result.returncode, [args[2:4] for args in installs]) == (
        0,
        [["--python", ".venv/bin/python"]],
    )


def test_complete_suite_uses_the_owned_interpreter(tmp_path):
    # Arrange
    (tmp_path / "tests").mkdir()
    script = _step("test", "Run pytest")
    # Act
    result, calls = _run(tmp_path, script)
    # Assert
    assert (result.returncode, calls) == (
        0,
        [["owned-python", ["-m", "pytest", "tests/", "-x"]]],
    )


def test_missing_suite_is_refused_before_invocation(tmp_path):
    # Arrange
    script = _step("test", "Run pytest")
    # Act
    result, calls = _run(tmp_path, script)
    # Assert
    assert (result.returncode, calls) == (1, [])


@pytest.mark.parametrize("port", ["", "0", "65536", "5432\nEVIL=1", "$(touch marker)"])
def test_invalid_service_port_cannot_export_a_store(tmp_path, port):
    # Arrange
    script = _step("test", "Point the store")
    # Act
    result, calls = _run(tmp_path, script, POSTGRES_SERVICE_PORT=port)
    # Assert
    assert (
        result.returncode,
        calls,
        (tmp_path / "github.env").exists(),
        (tmp_path / "marker").exists(),
    ) == (1, [], False, False)


def test_allocated_service_port_is_the_only_exported_store(tmp_path):
    # Arrange
    script = _step("test", "Point the store")
    # Act
    result, calls = _run(tmp_path, script)
    # Assert
    assert (result.returncode, calls, (tmp_path / "github.env").read_text()) == (
        0,
        [],
        "SCITEX_STORE_DSN=postgresql://scitex:scitex@127.0.0.1:32678/scitex\n",
    )


@pytest.mark.parametrize("ahead,existing", [("0", ""), ("1", "21"), ("1", "")])
def test_main_sync_only_opens_or_reuses_a_reviewable_pr(tmp_path, ahead, existing):
    # Arrange
    script = _step("sync-main", "Open")
    # Act
    result, calls = _run(tmp_path, script, AHEAD=ahead, EXISTING_PR=existing)
    merges = [
        arguments
        for name, arguments in calls
        if name == "gh" and arguments[:2] == ["pr", "merge"]
    ]
    creates = [
        arguments
        for name, arguments in calls
        if name == "gh" and arguments[:2] == ["pr", "create"]
    ]
    # Assert
    assert (result.returncode, merges, len(creates)) == (
        0,
        [],
        int(ahead == "1" and not existing),
    )
