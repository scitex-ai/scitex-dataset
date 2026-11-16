#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_corebench_inventory.py

"""CORE-Bench inventory helper: count a capsule tree's files by kind.

Split from :mod:`.corebench` (512-line-per-file guard).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# File-extension buckets for inventory file-counting.
_PY_EXTS = (".py",)
_R_EXTS = (".r", ".rmd")
_IPYNB_EXTS = (".ipynb",)
_DATA_EXTS = (
    ".csv",
    ".tsv",
    ".npy",
    ".h5",
    ".parquet",
    ".rds",
    ".rdata",
    ".mat",
    ".json",
    ".xlsx",
)


def _classify_capsule_files(capsule_dir: Path) -> Optional[dict]:
    """Walk a downloaded capsule directory and count files by extension."""
    if not capsule_dir.exists():
        return None
    n_py = n_r = n_ipynb = n_data = 0
    has_dockerfile = False
    for _root, _dirs, files in os.walk(capsule_dir):
        for f in files:
            fl = f.lower()
            if fl.endswith(_PY_EXTS):
                n_py += 1
            elif fl.endswith(_R_EXTS):
                n_r += 1
            elif fl.endswith(_IPYNB_EXTS):
                n_ipynb += 1
            elif fl in ("dockerfile",) or fl.startswith("dockerfile"):
                has_dockerfile = True
            elif fl.endswith(_DATA_EXTS):
                n_data += 1
    return {
        "n_python_files": n_py,
        "n_r_files": n_r,
        "n_notebooks": n_ipynb,
        "has_dockerfile": has_dockerfile,
        "n_data_files": n_data,
    }


__all__ = ["_classify_capsule_files"]

# EOF
