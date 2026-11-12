"""Real CLI and FastMCP reads must stay inside the bundled skill directory."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def skill_copy(tmp_path):
    root = tmp_path
    package = root / "src" / "scitex_dataset"
    source = Path(__file__).resolve().parents[2] / "src" / "scitex_dataset"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "_sphinx_html"))
    private = package / "_skills" / "owned-private.md"
    private.write_text("OWNED_BOUNDARY_MARKER", encoding="utf-8")
    pages = package / "_skills" / "scitex-dataset"
    (pages / "outside-link.md").symlink_to(private)
    return root, pages, private


def _read(interface, name, root):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(root / "src"), HOME=str(root),
               SCITEX_DIR=str(root / "state"), SCITEX_DATASET_CONFIG=" ")
    if interface == "cli":
        command = [sys.executable, "-m", "scitex_dataset", "skills", "get", name, "--json"]
    else:
        command = [sys.executable, "-c", (
            "import asyncio,sys; from fastmcp import FastMCP; "
            "from scitex_dataset._mcp._tools._skills import register_skills_tools; "
            "m=FastMCP('owned-skill-boundary'); register_skills_tools(m); "
            "r=asyncio.run(m.call_tool('skills_get', {'name':sys.argv[1]})); "
            "print(r.content[0].text)"
        ), name]
    result = subprocess.run(command, env=env, cwd=root, capture_output=True,
                            text=True, timeout=60, check=False)
    return result, json.loads(result.stdout)


@pytest.mark.parametrize("interface", ["cli", "mcp"])
@pytest.mark.parametrize("name", ["04_cli-reference", "SKILL"])
def test_named_bundled_page_keeps_exact_content(skill_copy, interface, name):
    # Arrange
    root, pages, _ = skill_copy
    expected = (pages / f"{name}.md").read_text(encoding="utf-8")
    # Act
    result, payload = _read(interface, name, root)
    # Assert
    assert (result.returncode, payload["success"], payload["content"]) == (0, True, expected)


@pytest.mark.parametrize("interface", ["cli", "mcp"])
@pytest.mark.parametrize("attack", ["parent", "absolute", "symlink", "backslash"])
def test_outside_resource_is_refused_without_reading_its_content(skill_copy, interface, attack):
    # Arrange
    root, _, private = skill_copy
    names = {"parent": "../owned-private", "absolute": str(private.with_suffix("")),
             "symlink": "outside-link", "backslash": "..\\owned-private"}
    expected_code = {"cli": 2, "mcp": 0}[interface]
    # Act
    result, payload = _read(interface, names[attack], root)
    # Assert
    assert (result.returncode, payload["success"], "OWNED_BOUNDARY_MARKER" in result.stdout) == (expected_code, False, False)
