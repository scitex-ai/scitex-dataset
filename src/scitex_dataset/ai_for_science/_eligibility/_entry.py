#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_entry.py

"""Parse a run script into the steps it executes (python / notebook / r / other).

Shell-level only: line continuations are joined, comments and ``VAR=``
prefixes dropped, ``&&`` / ``;`` / ``|`` split, and a called ``*.sh``
inside the archive is followed (depth 3). Commands that are not an
interpreter, notebook runner or script (``cd``, ``set``, ``pip``) are not
steps.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Iterable

from ._archive import _norm
from ._code import _SETUP_MODULES, _decode

# Python interpreter options that consume the next argument.
_PY_OPTS_WITH_VALUE = {"-W", "-X", "-Q"}


def _shell_commands(text: str) -> list[list[str]]:
    joined = re.sub(r"\\\r?\n", " ", text)
    commands: list[list[str]] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for part in re.split(r"&&|\|\||;|\|", line):
            try:
                tokens = shlex.split(part, comments=True)
            except ValueError:
                tokens = part.split()
            while tokens and re.match(r"^[A-Za-z_]\w*=", tokens[0]):
                tokens = tokens[1:]
            if tokens:
                commands.append(tokens)
    return commands


def _resolve(target: str, base_dir: str, names: set[str]) -> str | None:
    for cand in (_norm(posixpath.join(base_dir, target)), _norm(target.lstrip("/"))):
        if cand in names:
            return cand
    return None


def _python_step(args: list[str], base: str, names: set[str]):
    """``(kind, target)`` for one ``python ...`` command, or ``None`` for setup.

    ``python -m pip|venv|conda ...`` installs things and is not an analysis
    step; ``python -m <module>`` and ``python -c`` are; ``python [opts]
    script.py`` is, with the script resolved inside the archive.
    """
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m":
            mod = args[i + 1] if i + 1 < len(args) else ""
            return None if mod in _SETUP_MODULES else ("python", None)
        if arg == "-c":
            return ("python", None)
        if arg in _PY_OPTS_WITH_VALUE:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return ("python", _resolve(arg, base, names))
    return None


def _entry_steps(entry: str, texts: dict[str, bytes], names: set[str], depth: int = 0):
    """``[(kind, resolved_target_or_None, command)]`` executed by ``entry``."""
    base = posixpath.dirname(entry)
    steps: list[tuple[str, str | None, str]] = []
    for tokens in _shell_commands(_decode(texts.get(entry, b""))):
        prog = posixpath.basename(tokens[0])
        args = tokens[1:]
        cmd = " ".join(tokens)
        if re.fullmatch(r"python[\d.]*", prog):
            step = _python_step(args, base, names)
            if step is not None:
                steps.append((step[0], step[1], cmd))
        elif prog in ("jupyter", "papermill", "ipython") or prog.startswith("jupyter-"):
            nbs = [a for a in args if a.endswith(".ipynb")]
            if prog == "papermill":
                nbs = nbs[:1]  # input notebook; the second is the output copy
            executes = (
                prog != "jupyter"
                or "--execute" in args
                or "execute" in args
                or "run" in args
            )
            for nb in nbs if executes else []:
                steps.append(("notebook", _resolve(nb, base, names), cmd))
        elif prog in ("Rscript", "R"):
            steps.append(("r", None, cmd))
        elif prog in ("bash", "sh") or prog.endswith(".sh"):
            script = tokens[0] if prog.endswith(".sh") else next(
                (a for a in args if not a.startswith("-")), None)  # fmt: skip
            target = _resolve(script, base, names) if script else None
            if target and target in texts and depth < 3:
                steps.extend(_entry_steps(target, texts, names, depth + 1))
            else:
                steps.append(("other", target, cmd))
    return steps


def _entry_files(names: list[str], entry_scripts: Iterable[str] | None) -> list[str]:
    if entry_scripts is None:
        return []
    present = set(names)
    return [e for e in entry_scripts if e in present]  # caller's preference order


__all__ = ["_entry_files", "_entry_steps"]

# EOF
