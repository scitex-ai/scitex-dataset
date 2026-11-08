"""Exercise real workflow shell control flow with inert command fixtures.

These fixtures verify command selection and failure propagation. They do not
qualify dependency resolution, scientific tests, publication or branch reviews.
"""

import json
import os
from pathlib import Path
import subprocess

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

name, arguments = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as stream:
    stream.write(json.dumps([name, arguments]) + "\n")
if name in {"python", "pip"}:
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
    for name in ["python", "pip", "git", "gh"]:
        (directory / name).symlink_to(command.name)
    log = tmp_path / "commands.jsonl"
    environment = dict(os.environ)
    environment.update(
        PATH=str(directory) + os.pathsep + os.defpath,
        COMMAND_LOG=str(log),
        REFUSE_PROFILE="0",
        AHEAD="1",
        EXISTING_PR="21",
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
    calls = [json.loads(line) for line in log.read_text().splitlines()]
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
        [("python", ["-m", "pip", "install", "-e", ".[all,dev]"])],
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
