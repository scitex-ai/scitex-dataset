"""Check release identity with real owned Git commits and tags."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dataset_release_version", ROOT / ".github/ci/check_release_version.py"
)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


@pytest.fixture
def repository(tmp_path):
    # Arrange an owned repository without inherited hooks or signing.
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Owned release fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *args,
            ],
            cwd=tmp_path,
            text=True,
        ).strip()

    git("init", "--initial-branch=main")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.8.2"\n')
    git("add", "pyproject.toml")
    git("commit", "-m", "Owned release fixture")
    git("tag", "v0.8.2")
    return tmp_path, git


def test_release_tag_version_and_checkout_agree(repository):
    # Arrange
    root, git = repository
    # Act
    observed = GUARD.check_release(root, "v0.8.2")
    # Assert
    assert observed == git("rev-parse", "HEAD")


@pytest.mark.parametrize("tag", ["v0.8.1", "refs/heads/main"])
def test_wrong_version_or_branch_is_refused_before_build(repository, tag):
    # Arrange
    root, _ = repository
    # Act
    # Assert
    with pytest.raises(ValueError, match="project version"):
        GUARD.check_release(root, tag)


def test_a_later_checkout_cannot_publish_an_older_tag(repository):
    # Arrange
    root, git = repository
    (root / "change.txt").write_text("new source")
    git("add", "change.txt")
    git("commit", "-m", "Another commit")
    # Act
    # Assert
    with pytest.raises(ValueError, match="Checkout"):
        GUARD.check_release(root, "v0.8.2")


def _workflow():
    return yaml.safe_load(
        (
            ROOT / ".github/workflows/pypi-publish-and-github-release-on-tag.yml"
        ).read_text()
    )


@pytest.mark.parametrize("job", ["test", "build"])
def test_release_identity_is_checked_before_installation(job):
    # Arrange
    steps = _workflow()["jobs"][job]["steps"]
    # Act
    guard = next(
        s for s in steps if s.get("name") == "Verify release version and checkout"
    )
    install = next(s for s in steps if s.get("name", "").startswith("Install"))
    # Assert
    assert steps.index(guard) < steps.index(install)


@pytest.mark.parametrize("job", ["test", "build"])
def test_release_identity_receives_the_admitted_tag_as_data(job):
    # Arrange
    steps = _workflow()["jobs"][job]["steps"]
    # Act
    guard = next(
        s for s in steps if s.get("name") == "Verify release version and checkout"
    )
    # Assert
    assert guard["env"]["RELEASE_TAG"] == "${{ steps.ver.outputs.version }}"


@pytest.mark.parametrize("job", ["test", "build"])
def test_release_identity_executes_the_source_guard(job):
    # Arrange
    steps = _workflow()["jobs"][job]["steps"]
    # Act
    guard = next(
        s for s in steps if s.get("name") == "Verify release version and checkout"
    )
    # Assert
    assert (
        guard["run"]
        == 'python .github/ci/check_release_version.py --tag "$RELEASE_TAG"'
    )


def test_release_uses_an_allocated_postgres_service():
    # Arrange
    workflow = _workflow()
    # Act
    service = workflow["jobs"]["test"]["services"]["postgres"]
    # Assert
    assert (service["image"], service["ports"]) == ("postgres:16", [5432])


def test_release_store_handoff_reads_the_service_mapping():
    # Arrange
    steps = _workflow()["jobs"]["test"]["steps"]
    # Act
    handoff = next(
        s for s in steps if s.get("name") == "Point the store at the owned service"
    )
    # Assert
    assert (
        handoff["env"]["POSTGRES_SERVICE_PORT"]
        == "${{ job.services.postgres.ports['5432'] }}"
    )
