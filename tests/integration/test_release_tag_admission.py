"""Execute the real release resolver with untrusted input in an owned directory."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/pypi-publish-and-github-release-on-tag.yml"
)


def _resolve(tmp_path, job, event, requested, ref):
    step = next(
        s
        for s in yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]
        if s.get("id") == "ver"
    )
    values = {"github.event_name": event, "inputs.version": requested}

    def render(text):
        for key, value in values.items():
            text = text.replace("${{ " + key + " }}", value)
        return text

    output = tmp_path / "output"
    environment = dict(os.environ)
    environment.update(GITHUB_REF=ref, GITHUB_OUTPUT=str(output))
    environment.update(
        {key: render(value) for key, value in step.get("env", {}).items()}
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", render(step["run"])],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result, output.read_text() if output.exists() else ""


@pytest.mark.parametrize("job", ["test", "build"])
@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_valid_tag_is_the_only_output(tmp_path, job, event):
    # Arrange
    request = (job, event, "v0.8.2", "refs/tags/v0.8.2")
    # Act
    result, output = _resolve(tmp_path, *request)
    # Assert
    assert (result.returncode, output) == (0, "version=v0.8.2\n")


@pytest.mark.parametrize("job", ["test", "build"])
@pytest.mark.parametrize(
    "value",
    [
        "develop",
        "a" * 40,
        "v00.8.2",
        "v0.8.2-alpha",
        "v0.8.2\nextra=value",
        "",
    ],
)
def test_manual_nonrelease_refs_and_extra_outputs_are_refused(tmp_path, job, value):
    # Arrange
    request = (job, "workflow_dispatch", value, "refs/heads/main")
    # Act
    result, output = _resolve(tmp_path, *request)
    # Assert
    assert result.returncode != 0 and output == ""


@pytest.mark.parametrize("job", ["test", "build"])
def test_manual_input_is_data_and_cannot_execute_shell_code(tmp_path, job):
    # Arrange
    # The only possible side effect is inside this test's own directory.
    marker = tmp_path / "executed"
    value = 'v0.8.2"; touch ' + shlex.quote(str(marker)) + "; #"
    # Act
    result, output = _resolve(
        tmp_path, job, "workflow_dispatch", value, "refs/heads/main"
    )
    # Assert
    assert (marker.exists(), result.returncode == 0, output) == (False, False, "")


@pytest.mark.parametrize("job", ["test", "build"])
@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", "refs/heads/main"),
        ("pull_request", "refs/tags/v0.8.2"),
    ],
)
def test_non_tag_push_and_unexpected_event_are_refused(tmp_path, job, event, ref):
    # Arrange
    request = (job, event, "v0.8.2", ref)
    # Act
    result, output = _resolve(tmp_path, *request)
    # Assert
    assert result.returncode != 0 and output == ""


@pytest.mark.parametrize("job", ["test", "build", "release"])
def test_release_checkout_cannot_select_a_same_named_branch(job):
    # Arrange
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]
    # Act
    refs = [s["with"]["ref"] for s in steps if s.get("uses") == "actions/checkout@v4"]
    # Assert
    assert len(refs) == 1 and refs[0].startswith("refs/tags/${{")
