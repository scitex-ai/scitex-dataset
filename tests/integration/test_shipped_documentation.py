"""Check documentation identity in a genuinely uninstalled interpreter."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated_configuration(tmp_path):
    """Read the real config without site packages or the checkout as cwd."""
    code = (
        "import json,runpy,sys; "
        "config=runpy.run_path(sys.argv[1]); "
        "print(json.dumps({'release':config['release'],'root':sys.path[0]}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(ROOT / "docs/sphinx/conf.py")],
        cwd=tmp_path,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_uninstalled_documentation_uses_declared_source_version(isolated_configuration):
    # Arrange
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    # Act
    observed = isolated_configuration["release"]
    # Assert
    assert observed == expected


def test_documentation_import_root_is_independent_of_cwd(isolated_configuration):
    # Arrange
    expected = str(ROOT / "src")
    # Act
    observed = isolated_configuration["root"]
    # Assert
    assert observed == expected


def test_shipped_documentation_identifies_its_source_version():
    # Arrange
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    options = ROOT / "src/scitex_dataset/_sphinx_html/_static/documentation_options.js"
    # Act
    observed = re.search(r"\bVERSION:\s*(['\"])([^'\"]+)\1", options.read_text())
    # Assert
    assert observed is not None and observed.group(2) == expected


def test_shipped_documentation_has_only_tracked_build_inputs():
    # Arrange
    bundle = ROOT / "src/scitex_dataset/_sphinx_html"
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "--", "src/scitex_dataset/_sphinx_html"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    )
    # Act
    untracked = {
        str(path.relative_to(ROOT))
        for path in bundle.rglob("*")
        if path.is_file() and str(path.relative_to(ROOT)) not in tracked
    }
    # Assert
    assert not untracked, sorted(untracked)


@pytest.mark.parametrize(
    "command",
    [
        "neuroscience-openneuro-fetch",
        "biology-geo-fetch",
        "completion-install",
    ],
)
def test_shipped_cli_reference_contains_rendered_commands(command):
    # Arrange
    reference = ROOT / "src/scitex_dataset/_sphinx_html/cli_reference.html"
    # Act
    anchors = set(re.findall(r'id="([^"]+)"', reference.read_text()))
    # Assert
    assert "scitex-dataset-" + command in anchors
