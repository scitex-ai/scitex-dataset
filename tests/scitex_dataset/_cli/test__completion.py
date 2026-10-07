#!/usr/bin/env python3
"""Completion drop-in v1 — isolated HOME/SCITEX_DIR tests.

Every install test runs under an isolated temp HOME (plus an explicit
$SCITEX_DIR) so the real ~/.scitex, ~/.bashrc and ~/.zshrc are never
touched. Env isolation hand-swaps os.environ (no monkeypatch/mocker)
and restores the exact snapshot on teardown.

House style: one assertion per test, explicit Arrange/Act/Assert markers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# click is a runtime dependency of scitex-dataset.
CliRunner = pytest.importorskip("click.testing").CliRunner

from scitex_dataset._cli import main
from scitex_dataset._cli._completion import dropin_path


@pytest.fixture()
def isolated_home(tmp_path):
    """Point HOME + SCITEX_DIR at tmp_path; return (home, scitex_dir)."""
    # Arrange
    home = tmp_path / "home"
    # Act
    home.mkdir()
    scitex_dir = tmp_path / "scitex-root"
    old_home = os.environ.get("HOME")
    old_scitex = os.environ.get("SCITEX_DIR")
    os.environ["HOME"] = str(home)
    os.environ["SCITEX_DIR"] = str(scitex_dir)
    try:
        yield home, scitex_dir
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_scitex is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = old_scitex


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


def _expected_dropin(scitex_dir: Path) -> Path:
    return scitex_dir / "dataset" / "runtime" / "completion" / "scitex-dataset"


def _install_bash_yes():
    return _invoke("completion", "install", "--shell", "bash", "--yes")


# ----- install writes the drop-in and prints the path ------------------ #


def test_completion_install_exit_code(isolated_home):
    # Arrange
    # Act
    result = _install_bash_yes()
    # Assert
    assert result.exit_code == 0


def test_completion_install_dropin_is_file(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    # Act
    target = _expected_dropin(scitex_dir)
    # Assert
    assert target.is_file()


def test_completion_install_stdout_prints_path(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    # Act
    result = _install_bash_yes()
    # Assert
    assert str(_expected_dropin(scitex_dir)) in result.output


def test_completion_install_dropin_nonempty(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    # Act
    target = _expected_dropin(scitex_dir)
    # Assert
    assert target.stat().st_size > 0


# ----- install is idempotent ------------------------------------------- #


def test_completion_install_second_run_exit_code(isolated_home):
    # Arrange
    _install_bash_yes()
    # Act
    result = _install_bash_yes()
    # Assert
    assert result.exit_code == 0


def test_completion_install_second_run_same_content(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    target = _expected_dropin(scitex_dir)
    before = target.read_text(encoding="utf-8")
    # Act
    _install_bash_yes()
    # Assert
    assert target.read_text(encoding="utf-8") == before


def test_completion_install_second_run_prints_path(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    # Act
    result = _install_bash_yes()
    # Assert
    assert str(_expected_dropin(scitex_dir)) in result.output


# ----- $SCITEX_DIR is honoured ----------------------------------------- #


def test_completion_install_honors_scitex_dir_exit_code(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom-scitex"
    old_home = os.environ.get("HOME")
    old_scitex = os.environ.get("SCITEX_DIR")
    os.environ["HOME"] = str(home)
    os.environ["SCITEX_DIR"] = str(custom)
    try:
        # Act
        result = _invoke("completion", "install", "--shell", "bash")
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_scitex is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = old_scitex
    # Assert
    assert result.exit_code == 0


def test_completion_install_honors_scitex_dir_dropin(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom-scitex"
    old_home = os.environ.get("HOME")
    old_scitex = os.environ.get("SCITEX_DIR")
    os.environ["HOME"] = str(home)
    os.environ["SCITEX_DIR"] = str(custom)
    try:
        # Act
        _invoke("completion", "install", "--shell", "bash")
        target = custom / "dataset" / "runtime" / "completion" / "scitex-dataset"
        exists = target.is_file()
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_scitex is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = old_scitex
    # Assert
    assert exists


def test_completion_install_defaults_to_home_scitex_dropin(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    old_home = os.environ.get("HOME")
    old_scitex = os.environ.get("SCITEX_DIR")
    os.environ["HOME"] = str(home)
    os.environ.pop("SCITEX_DIR", None)
    try:
        # Act
        result = _invoke("completion", "install", "--shell", "bash")
        target = (
            home / ".scitex" / "dataset" / "runtime" / "completion" / "scitex-dataset"
        )
        exists = target.is_file()
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_scitex is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = old_scitex
    # Assert
    assert result.exit_code == 0


def test_completion_install_default_path_under_home(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    old_home = os.environ.get("HOME")
    old_scitex = os.environ.get("SCITEX_DIR")
    os.environ["HOME"] = str(home)
    os.environ.pop("SCITEX_DIR", None)
    try:
        # Act
        _invoke("completion", "install", "--shell", "bash")
        target = (
            home / ".scitex" / "dataset" / "runtime" / "completion" / "scitex-dataset"
        )
        exists = target.is_file()
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_scitex is None:
            os.environ.pop("SCITEX_DIR", None)
        else:
            os.environ["SCITEX_DIR"] = old_scitex
    # Assert
    assert exists


# ----- rc files are never touched -------------------------------------- #


def test_completion_install_bashrc_unchanged(isolated_home):
    # Arrange
    home, _ = isolated_home
    bashrc = home / ".bashrc"
    bashrc.write_text("# pristine\n", encoding="utf-8")
    before = bashrc.read_text(encoding="utf-8")
    # Act
    _install_bash_yes()
    # Assert
    assert bashrc.read_text(encoding="utf-8") == before


def test_completion_install_zshrc_unchanged(isolated_home):
    # Arrange
    home, _ = isolated_home
    zshrc = home / ".zshrc"
    zshrc.write_text("# pristine\n", encoding="utf-8")
    before = zshrc.read_text(encoding="utf-8")
    # Act
    _install_bash_yes()
    # Assert
    assert zshrc.read_text(encoding="utf-8") == before


def test_completion_install_exit_code_with_pristine_rc(isolated_home):
    # Arrange
    home, _ = isolated_home
    (home / ".bashrc").write_text("# pristine\n", encoding="utf-8")
    (home / ".zshrc").write_text("# pristine\n", encoding="utf-8")
    # Act
    result = _install_bash_yes()
    # Assert
    assert result.exit_code == 0


# ----- status checks the drop-in --------------------------------------- #


def test_completion_status_missing_reports_not_installed(isolated_home):
    # Arrange
    # Act
    result = _invoke("completion", "status")
    # Assert
    assert "not installed" in result.output.lower()


def test_completion_status_after_install_exit_code(isolated_home):
    # Arrange
    _install_bash_yes()
    # Act
    result = _invoke("completion", "status")
    # Assert
    assert result.exit_code == 0


def test_completion_status_after_install_prints_path(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    # Act
    result = _invoke("completion", "status")
    # Assert
    assert str(_expected_dropin(scitex_dir)) in result.output


def test_completion_status_after_install_says_installed(isolated_home):
    # Arrange
    _install_bash_yes()
    # Act
    result = _invoke("completion", "status")
    # Assert
    assert "installed" in result.output.lower()


def test_completion_status_json_exit_code(isolated_home):
    # Arrange
    _install_bash_yes()
    # Act
    result = _invoke("completion", "status", "--json")
    # Assert
    assert result.exit_code == 0


def test_completion_status_json_installed_true(isolated_home):
    # Arrange
    _install_bash_yes()
    # Act
    payload = json.loads(_invoke("completion", "status", "--json").output)
    # Assert
    assert payload["installed"] is True


def test_completion_status_json_path(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _install_bash_yes()
    # Act
    payload = json.loads(_invoke("completion", "status", "--json").output)
    # Assert
    assert payload["path"] == str(_expected_dropin(scitex_dir))


# ----- shims -------------------------------------------------------- #


def test_install_shell_completion_shim_exit_code(isolated_home):
    # Arrange
    # Act
    result = _invoke("install-shell-completion", "--shell", "bash")
    # Assert
    assert result.exit_code == 0


def test_install_shell_completion_shim_writes_dropin(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    _invoke("install-shell-completion", "--shell", "bash")
    # Act
    target = _expected_dropin(scitex_dir)
    # Assert
    assert target.is_file()


def test_install_shell_completion_shim_prints_path(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    # Act
    result = _invoke("install-shell-completion", "--shell", "bash")
    # Assert
    assert str(_expected_dropin(scitex_dir)) in result.output


def test_print_shell_completion_exit_code():
    # Arrange
    # Act
    result = _invoke("print-shell-completion", "--shell", "bash")
    # Assert
    assert result.exit_code == 0


def test_print_shell_completion_nonempty_stdout():
    # Arrange
    # Act
    result = _invoke("print-shell-completion", "--shell", "bash")
    # Assert
    assert result.output.strip()


# ----- source hygiene --------------------------------------------------- #


def test_completion_source_has_no_builtin_print():
    # Arrange
    src = Path(__file__).resolve().parents[3] / "src" / "scitex_dataset" / "_cli"
    # Act
    combined = "".join(
        (src / name).read_text(encoding="utf-8")
        for name in ("_completion.py", "__init__.py")
    )
    # Assert
    assert "print(" not in combined


def test_completion_source_has_no_rc_append_path():
    # Arrange
    src = Path(__file__).resolve().parents[3] / "src" / "scitex_dataset" / "_cli"
    # Act
    text = (src / "_completion.py").read_text(encoding="utf-8")
    # Assert
    assert '".bashrc"' not in text


def test_completion_source_has_no_zshrc_append_path():
    # Arrange
    src = Path(__file__).resolve().parents[3] / "src" / "scitex_dataset" / "_cli"
    # Act
    text = (src / "_completion.py").read_text(encoding="utf-8")
    # Assert
    assert '".zshrc"' not in text


def test_dropin_path_respects_scitex_dir_env(isolated_home):
    # Arrange
    _, scitex_dir = isolated_home
    # Act
    resolved = dropin_path()
    # Assert
    assert resolved == _expected_dropin(scitex_dir)
