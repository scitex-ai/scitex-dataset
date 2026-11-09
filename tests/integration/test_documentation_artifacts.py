"""Real ZIP/tar fixtures cover the prepublication documentation byte gate."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "documentation_artifacts", ROOT / ".github/ci/check_documentation_artifacts.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def docs_tree(tmp_path):
    contents = {
        "index.html": b'<img src="_static/icon.svg">',
        "_static/icon.svg": b"<svg></svg>",
        "_sources/index.rst.txt": b"Dataset documentation",
    }
    bundle = tmp_path / "src/scitex_dataset/_sphinx_html"
    for name, data in contents.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    dist = tmp_path / "dist"
    dist.mkdir()
    return tmp_path, dist, contents


def _archives(dist, wheel_files, source_files):
    with zipfile.ZipFile(dist / "scitex_dataset-0.8.3-py3-none-any.whl", "w") as wheel:
        for name, data in wheel_files.items():
            wheel.writestr("scitex_dataset/_sphinx_html/" + name, data)
    with tarfile.open(dist / "scitex_dataset-0.8.3.tar.gz", "w:gz") as source:
        for name, data in source_files.items():
            info = tarfile.TarInfo(
                "scitex_dataset-0.8.3/src/scitex_dataset/_sphinx_html/" + name
            )
            info.size = len(data)
            source.addfile(info, io.BytesIO(data))


def _rejection(root, dist):
    try:
        CHECK.check(root, dist)
    except ValueError as error:
        return str(error)
    return None


def test_complete_documentation_archives_are_accepted(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(dist, contents, contents)
    # Act
    result = CHECK.check(root, dist)
    # Assert
    assert result["public_docs_files"] == 3


def test_missing_wheel_icon_is_refused(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(
        dist, {k: v for k, v in contents.items() if k != "_static/icon.svg"}, contents
    )
    # Act
    error = _rejection(root, dist)
    # Assert
    assert error == "Wheel documentation missing or changed: _static/icon.svg"


def test_missing_source_copy_is_refused(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(
        dist,
        contents,
        {k: v for k, v in contents.items() if k != "_sources/index.rst.txt"},
    )
    # Act
    error = _rejection(root, dist)
    # Assert
    assert (
        error
        == "Source archive documentation missing or changed: _sources/index.rst.txt"
    )


def test_changed_wheel_bytes_are_refused(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(dist, dict(contents, **{"index.html": b"changed"}), contents)
    # Act
    error = _rejection(root, dist)
    # Assert
    assert error == "Wheel documentation missing or changed: index.html"


def test_changed_source_bytes_are_refused(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(dist, contents, dict(contents, **{"index.html": b"changed"}))
    # Act
    error = _rejection(root, dist)
    # Assert
    assert error == "Source archive documentation missing or changed: index.html"


def test_duplicate_wheel_members_are_refused(docs_tree):
    # Arrange
    root, dist, contents = docs_tree
    _archives(dist, contents, contents)
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "scitex_dataset/_sphinx_html/index.html", contents["index.html"]
        )
    # Act
    error = _rejection(root, dist)
    # Assert
    assert error == "Duplicate wheel members are not admitted"


def test_missing_source_documentation_cannot_pass_vacuously(tmp_path):
    # Arrange
    dist = tmp_path / "dist"
    dist.mkdir()
    # Act
    error = _rejection(tmp_path, dist)
    # Assert
    assert error == "Source documentation directory is missing or symbolic"


def test_artifact_gate_precedes_upload_and_publication():
    # Arrange
    workflow = yaml.safe_load(
        (
            ROOT / ".github/workflows/pypi-publish-and-github-release-on-tag.yml"
        ).read_text()
    )
    steps = workflow["jobs"]["build"]["steps"]
    # Act
    build = next(
        i for i, s in enumerate(steps) if s.get("name") == "Build distribution"
    )
    check = next(
        i
        for i, s in enumerate(steps)
        if s.get("name") == "Verify built documentation against source"
    )
    upload = next(
        i
        for i, s in enumerate(steps)
        if s.get("uses", "").startswith("actions/upload-artifact@")
    )
    # Assert
    assert build < check < upload
