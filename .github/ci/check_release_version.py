"""Stdlib-only release boundary, before installing the selected checkout."""

import argparse
import subprocess
from pathlib import Path

import tomllib


def check_release(root: Path, tag: str) -> str:
    """Require the requested tag, project version and checked-out commit to agree."""
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    if not isinstance(version, str) or tag != f"v{version}":
        raise ValueError("Release tag does not match the project version")

    def revision(ref: str) -> str:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", ref], cwd=root, text=True
        ).strip()

    commit = revision(f"refs/tags/{tag}^{{commit}}")
    if commit != revision("HEAD^{commit}"):
        raise ValueError("Checkout does not match the requested release tag")
    return commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    arguments = parser.parse_args()
    commit = check_release(Path.cwd(), arguments.tag)
    print(f"Release source: {arguments.tag} at {commit}")
