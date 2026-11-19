#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_facts.py

"""Dataset-blind eligibility facts read from a capsule's file tree.

Every benchmark adapter hands this module the same thing — the member
paths and the small text files of ONE unit's archive — and gets the same
fact record back. Nothing here knows a benchmark: the environment spec is
any ``Dockerfile`` / ``requirements*.txt`` / ``environment.y(a)ml`` /
``postInstall`` in the tree, the code is every ``.py`` / ``.ipynb`` /
``.R`` file, and the entry is either the run script(s) the adapter names
or, for archives without one, every notebook and script.

Facts (all JSON-serialisable; lists are sorted):

- ``file_languages``      languages of code files by extension
- ``kernel_languages``    languages declared by notebook kernels
- ``magic_languages``     languages run from Python: notebook magics
                          (``%%R``, ``%R``, ``%load_ext rpy2.ipython``) or ``import rpy2``
- ``toolchain_languages`` languages installed by the environment spec or
                          invoked by the entry (``r-base``, ``Rscript``)
- ``code_languages``      union of the four above
- ``dl_frameworks``       deep-learning frameworks installed or imported
                          (``tensorflow``, ``torch``, ``keras``, ``jax``, ``mxnet``)
- ``gpu_in_environment``  the environment spec asks for a GPU (CUDA/cuDNN/
                          TensorRT base image, ``cudatoolkit``, ``*-gpu``, ``+cuNNN``)
- ``gpu_in_code``         code uses a GPU (``cupy``, ``.cuda()``, ``"cuda"`` device)
- ``gpu_required``        either of the two
- ``entry_kinds``         what the entry executes: ``python``, ``notebook``,
                          ``r``, ``other``
- ``n_entry_steps``       number of executed scripts/notebooks/modules
- ``entry_notebook_code_cells``    max non-empty code cells of an entry notebook
- ``entry_notebook_local_imports`` max project-local modules an entry
                                   notebook imports
- ``evidence``            the matched lines/paths behind each fact (audit only)

No answer, oracle or result file is read: only member names, environment
specs, run scripts and code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ._code import (
    _CODE_EXT,
    _FRAMEWORK_IMPORTS,
    _GPU_CODE,
    _MAGIC_LANG,
    _MAGIC_LINE,
    _code_cells,
    _decode,
    _env_facts,
    _imports,
    _kernel_language,
    _local_modules,
    _notebook,
    _python_source_of_cells,
)
from ._entry import _entry_files, _entry_steps


# ---------------------------------------------------------------------------
# Public: one fact record per unit tree
# ---------------------------------------------------------------------------


def analyze_tree(
    names: list[str],
    texts: dict[str, bytes],
    *,
    entry_scripts: Iterable[str] | None = None,
) -> dict:
    """Derive the archive-stage facts of one unit (see module docstring).

    ``entry_scripts`` names the run script(s) (e.g. ``["code/run",
    "code/run.sh"]``); the first that exists is parsed. ``None`` means the
    archive has no run script and every notebook/script is an entry.
    """
    name_set = set(names)
    file_langs = {_CODE_EXT[Path(n).suffix.lower()] for n in names
                  if Path(n).suffix.lower() in _CODE_EXT}  # fmt: skip
    kernel_langs: set[str] = set()
    magic_langs: set[str] = set()
    magic_ev: list[str] = []
    code_sources: list[str] = []
    notebooks: dict[str, dict] = {}
    gpu_ev: list[str] = []
    for rel in sorted(texts):
        low = rel.lower()
        if low.endswith(".ipynb"):
            nb = _notebook(texts[rel])
            if nb is None:
                continue
            notebooks[rel] = nb
            lang = _kernel_language(nb)
            if lang:
                kernel_langs.add(lang)
            cells = _code_cells(nb)
            for src in cells:  # %%R / %R magics and the rpy2 bridge, any line
                for m in _MAGIC_LINE.finditer(src):
                    name = m.group(1).lower()
                    if name in _MAGIC_LANG:
                        magic_langs.add(_MAGIC_LANG[name])
                        magic_ev.append(f"{rel}: {m.group(0).strip()}")
            if lang in (None, "python"):
                code_sources.append(_python_source_of_cells(cells))
        elif low.endswith(".py"):
            code_sources.append(_decode(texts[rel]))
    env = _env_facts(texts)
    frameworks = set(env["frameworks"])
    fw_ev = list(env["evidence"]["dl_frameworks"])
    for src in code_sources:
        for mod in _imports(src):
            if mod in _FRAMEWORK_IMPORTS:
                frameworks.add(_FRAMEWORK_IMPORTS[mod])
                fw_ev.append(f"import {mod}")
            elif mod == "rpy2":  # R driven from Python
                magic_langs.add("r")
                magic_ev.append("import rpy2")
        m = _GPU_CODE.search(src)
        if m:
            gpu_ev.append(f"code: {m.group(0).strip()}")
    toolchain = set(env["toolchain"])

    entries = _entry_files(names, entry_scripts)
    if entry_scripts is None:
        steps = [("notebook", n, n) for n in sorted(notebooks)]
        steps += [("python", n, n) for n in names if n.endswith(".py")]
    elif entries:
        steps = _entry_steps(entries[0], texts, name_set)
    else:
        steps = []
    if any(kind == "r" for kind, _, _ in steps):
        toolchain.add("r")
    local = _local_modules(names)
    nb_cells, nb_local = 0, 0
    for kind, target, _ in steps:
        if kind == "notebook" and target in notebooks:
            cells = [c for c in _code_cells(notebooks[target]) if c.strip()]
            nb_cells = max(nb_cells, len(cells))
            nb_local = max(
                nb_local, len(_imports(_python_source_of_cells(cells)) & local)
            )
    return {
        "file_languages": sorted(file_langs),
        "kernel_languages": sorted(kernel_langs),
        "magic_languages": sorted(magic_langs),
        "toolchain_languages": sorted(toolchain),
        "code_languages": sorted(file_langs | kernel_langs | magic_langs | toolchain),
        "dl_frameworks": sorted(frameworks),
        "gpu_in_environment": bool(env["evidence"]["gpu"]),
        "gpu_in_code": bool(gpu_ev),
        "gpu_required": bool(env["evidence"]["gpu"] or gpu_ev),
        "entry_kinds": sorted({kind for kind, _, _ in steps}),
        "n_entry_steps": len(steps),
        "entry_notebook_code_cells": nb_cells,
        "entry_notebook_local_imports": nb_local,
        "evidence": {
            "entry": entries[:1],
            "entry_steps": [f"{kind}: {cmd}" for kind, _, cmd in steps],
            "gpu": env["evidence"]["gpu"] + gpu_ev,
            "dl_frameworks": sorted(set(fw_ev)),
            "toolchain": env["evidence"]["toolchain"],
            "magic": magic_ev,
        },
    }


__all__ = ["analyze_tree"]

# EOF
