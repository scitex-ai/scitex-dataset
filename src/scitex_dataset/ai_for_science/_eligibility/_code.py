#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_code.py

"""Code and environment-spec readers shared by every adapter (no benchmark knowledge)."""

from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Iterable

from ._archive import _ENV_NAME

_CODE_EXT = {".py": "python", ".r": "r", ".rmd": "r", ".jl": "julia"}
# Package-name prefixes -> canonical deep-learning framework.
_FRAMEWORK_PREFIXES = (
    ("tensorflow", "tensorflow"),
    ("tf-nightly", "tensorflow"),
    ("pytorch", "torch"),
    ("torch", "torch"),
    ("keras", "keras"),
    ("jaxlib", "jax"),
    ("jax", "jax"),
    ("flax", "jax"),
    ("mxnet", "mxnet"),
)
# Top-level import names -> framework.
_FRAMEWORK_IMPORTS = {
    "tensorflow": "tensorflow",
    "torch": "torch",
    "torchvision": "torch",
    "torchaudio": "torch",
    "torch_geometric": "torch",
    "keras": "keras",
    "jax": "jax",
    "jaxlib": "jax",
    "flax": "jax",
    "mxnet": "mxnet",
}
_GPU_IMAGE = re.compile(r"cuda|cudnn|tensorrt|nvidia|(^|[^a-z])gpu", re.I)
_GPU_ENV = re.compile(
    r"(^|[\s=])(cudatoolkit|cuda-[\w.-]+|cupy[\w-]*|nvidia-[\w-]+|[\w.-]+-gpu)([=<>\s]|$)|\+cu\d{2,3}",
    re.I,
)
_GPU_CODE = re.compile(
    r"^\s*(import|from)\s+cupy\b|\.cuda\(\)|device\(\s*['\"]cuda|['\"]cuda(:\d+)?['\"]"
    # CUDA_VISIBLE_DEVICES is deliberately absent: it is as often set to ""
    # or "-1" to FORCE the CPU as to pick a GPU.
    r"|list_physical_devices\(\s*['\"]GPU",
    re.M,
)
_R_ENV = re.compile(
    r"(^|[\s=])(r-base|r-[a-z0-9.]+|r-cran-[\w.-]+)([=<>\s]|$)|\bRscript\b|install\.packages\(|BiocManager",
    re.M,
)
_KERNEL_LANG = {"ir": "r", "r": "r", "python": "python", "python3": "python",
                "python2": "python", "julia": "julia"}  # fmt: skip
_MAGIC_LANG = {"r": "r", "julia": "julia", "matlab": "matlab", "octave": "octave", "rpy2": "r"}
# ``%%R`` / ``%R`` magics and ``%load_ext rpy2.ipython`` on any line of a cell.
_MAGIC_LINE = re.compile(r"^\s*%%?(?:load_ext\s+)?([A-Za-z]\w*)", re.M)
_SETUP_MODULES = {"pip", "ensurepip", "venv", "virtualenv", "conda"}


# ---------------------------------------------------------------------------
# Code analysis helpers
# ---------------------------------------------------------------------------


def _decode(body: bytes) -> str:
    return body.decode("utf-8", "replace")


def _notebook(body: bytes) -> dict | None:
    try:
        nb = json.loads(_decode(body))
    except ValueError:
        return None
    return nb if isinstance(nb, dict) else None


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _code_cells(nb: dict) -> list[str]:
    cells = nb.get("cells") or []
    return [_cell_source(c) for c in cells
            if isinstance(c, dict) and c.get("cell_type") == "code"]  # fmt: skip


def _kernel_language(nb: dict) -> str | None:
    meta = nb.get("metadata") or {}
    spec = meta.get("kernelspec") or {}
    info = meta.get("language_info") or {}
    for value in (spec.get("language"), info.get("name"), spec.get("name")):
        if isinstance(value, str) and value:
            key = value.strip().lower()
            return _KERNEL_LANG.get(key, key.split("-")[0])
    return None


def _python_source_of_cells(cells: Iterable[str]) -> str:
    """Python text of notebook cells: cell magics skipped, line magics blanked."""
    lines: list[str] = []
    for src in cells:
        if src.lstrip().startswith("%%"):
            continue
        for line in src.splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
    return "\n".join(lines)


def _imports(source: str) -> set[str]:
    """Top-level module names imported by Python ``source`` (py2-tolerant)."""
    try:
        with warnings.catch_warnings():  # old code: invalid escapes etc.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        found = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", source, re.M)
        return set(found)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
    return mods


def _local_modules(names: Iterable[str]) -> set[str]:
    """Module/package names a project could import from its own files."""
    mods: set[str] = set()
    for name in names:
        if name.endswith(".py"):
            # The module stem and its package dirs; the archive's top
            # directory (``code/``) is the project root, not a package.
            parts = name[:-3].split("/")[1:] if "/" in name else [name[:-3]]
            mods.update(p for p in parts if p.isidentifier())
    return mods


def _framework_of_package(token: str) -> str | None:
    name = re.split(r"[=<>~!;\[@ ]", token.strip().strip("'\"").lower(), 1)[0]
    name = name.replace("_", "-")
    for prefix, framework in _FRAMEWORK_PREFIXES:
        if (
            name == prefix
            or name.startswith(prefix + "-")
            or (prefix == "torch" and name.startswith("torch"))
        ):
            return framework
    return None


def _env_facts(texts: dict[str, bytes]) -> dict:
    frameworks: set[str] = set()
    toolchain: set[str] = set()
    evidence: dict[str, list[str]] = {"gpu": [], "dl_frameworks": [], "toolchain": []}
    for rel in sorted(texts):
        if not _ENV_NAME.search(rel.lower()):
            continue
        text = _decode(texts[rel])
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.upper().startswith("FROM "):
                image = stripped.split()[1] if len(stripped.split()) > 1 else ""
                if _GPU_IMAGE.search(image):
                    evidence["gpu"].append(f"{rel}: {stripped}")
                for seg in re.split(r"[/:]", image.lower()):
                    fw = _framework_of_package(seg)
                    if fw:
                        frameworks.add(fw)
                        evidence["dl_frameworks"].append(f"{rel}: {stripped}")
                continue
            if _GPU_ENV.search(stripped):
                evidence["gpu"].append(f"{rel}: {stripped}")
            if _R_ENV.search(stripped):
                toolchain.add("r")
                evidence["toolchain"].append(f"{rel}: {stripped}")
            for token in re.split(r"[\s\\]+", stripped):
                fw = _framework_of_package(token)
                if fw and not token.startswith(("-", "http", "/")):
                    frameworks.add(fw)
                    evidence["dl_frameworks"].append(f"{rel}: {stripped}")
    return {"frameworks": frameworks, "toolchain": toolchain, "evidence": evidence}


# Figure-reading questions, for benchmarks without an upstream marker.
FIGURE_QUESTION_PATTERN = re.compile(
    r"\b(fig(ure)?s?|plots?|plotted|panels?|images?|graphs?|charts?|heat ?maps?"
    r"|histograms?|x-axis|y-axis|axis)\b",
    re.I,
)


def is_figure_question(text: str) -> bool:
    """True when ``text`` asks about a figure, plot, image or panel."""
    return bool(FIGURE_QUESTION_PATTERN.search(text or ""))


__all__ = ["FIGURE_QUESTION_PATTERN", "is_figure_question"]

# EOF
