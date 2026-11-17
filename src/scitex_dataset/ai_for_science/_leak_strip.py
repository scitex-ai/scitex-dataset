#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_leak_strip.py

"""Answer-leak handling for extracted capsules (ADR 0001).

Structural strips (output/log dirs and files, executed-notebook outputs)
plus answer-value redaction and its fail-loud guard. Split from
``_standardize``.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from ._capsule_layout import _rmtree


class AnswerLeakError(RuntimeError):
    """Raised when the supported text scan still detects an answer match.

    This guard is bounded by its extensions, size limit and matching rules;
    it does not qualify arbitrary input files. See :func:`_assert_no_value_leak`.
    """


class InvalidReferenceSourceError(ValueError):
    """A supplied masking reference cannot be qualified; no outputs changed."""


# Top-level subdirectories inside an extracted capsule that ship the
# authors' ORIGINAL outputs — i.e. the answer the task asks the agent to
# reproduce (e.g. CoreBench's ``results/classification/log_evaluate.txt``
# holds the eval-loss). They are stripped from the agent-visible
# ``input/`` after extraction so the answer is never reachable; the
# canonical answers live only in the operator-side ``eval/answers.jsonl``
# (never mounted). Listed as a constant so the leak surface is auditable
# and extensible per benchmark. Author run outputs / running logs must
# NEVER ship in for_solver for ANY capsule or cohort, and they are stripped
# at ANY depth (often nested under code/, e.g. code/dump/, code/log/).
LEAK_DIRS = ("results", "result", "output", "outputs", "log", "logs", "dump", "dumps")

# Loose running-log FILES that live OUTSIDE LEAK_DIRS (e.g. a stray
# ``data/feature_..._log.csv``) — removed by :func:`_strip_leak_files`. A file
# matches when it ends ``.log`` OR its stem has a DELIMITED ``log`` segment
# (``(^|[_.-])log([_.-]|$)`` — so ``catalog`` / ``dialog`` / ``logger`` are
# NOT matched) AND its extension is a text/data artifact (so code/config like
# ``log_utils.py`` or ``logging.yaml`` is kept).
_LEAK_FILE_RE = re.compile(r"(^|[_.-])log([_.-]|$)", re.IGNORECASE)
_LEAK_FILE_EXTS = {".log", ".txt", ".out", ".csv", ".tsv", ".dat"}

# Text extensions scanned by the value-verify guard (:func:`_assert_no_value_leak`).
_SCAN_EXTS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".out",
    ".tex",
    ".html",
    ".rst",
    ".py",
    ".r",
    ".ipynb",
    ".dat",
    ".cfg",
    ".ini",
}


def _strip_leak_dirs(input_dir: Path) -> list[str]:
    """Remove answer-bearing dirs (:data:`LEAK_DIRS`) from ``input_dir`` at ANY depth.

    Walks the whole ``input/`` tree and recursively deletes every directory
    whose name matches :data:`LEAK_DIRS` (case-insensitive). Author run
    outputs / running logs are frequently nested under ``code/`` (e.g.
    ``input/code/dump/``, ``input/code/log/``), not just at the top level, so
    a top-level-only strip would miss them. Symlinked dirs are skipped (not
    followed). Returns the removed paths as strings (for the caller's report /
    tests).
    """
    removed: list[str] = []
    leak_names = {name.lower() for name in LEAK_DIRS}
    for root, dirs, _ in os.walk(input_dir, topdown=True):
        kept: list[str] = []
        for d in dirs:
            p = Path(root) / d
            if d.lower() in leak_names and not p.is_symlink():
                _rmtree(p)
                removed.append(str(p))
            else:
                kept.append(d)
        dirs[:] = kept  # don't descend into dirs we just removed
    return removed


def _strip_leak_files(input_dir: Path) -> list[str]:
    """Remove loose running-log FILES from ``input_dir`` at any depth.

    Catches author run-logs that live OUTSIDE the :data:`LEAK_DIRS` dirs
    (e.g. ``data/feature_and_set_selection_log.csv``) which the dir strip
    alone misses. A file is removed when it ends ``.log`` OR (its stem has a
    delimited ``log`` segment via :data:`_LEAK_FILE_RE` AND its extension is a
    text/data artifact in :data:`_LEAK_FILE_EXTS`, so code/config like
    ``log_utils.py`` is kept). Returns the removed paths as strings.
    """
    removed: list[str] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            is_log = ext == ".log" or (
                ext in _LEAK_FILE_EXTS and _LEAK_FILE_RE.search(stem) is not None
            )
            if not is_log:
                continue
            p = Path(root) / f
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:  # pragma: no cover - unlink race
                pass
    return removed


# Marker written in place of a leaked answer value when redacting a KEPT file.
_VALUE_REDACTION = "[redacted]"

# A numeric token: requires a decimal point so bare integers are never matched.
_NUM_TOKEN_RE = re.compile(r"[0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?")


def _is_value_scan_target(path: Path) -> bool:
    """A regular (non-symlink) file with a :data:`_SCAN_EXTS` extension, or an
    EXTENSIONLESS text file (``Dockerfile``, ``code/run``, ``code/evaluation``).

    Symlinks are never followed, so an in-place redaction can never write
    through a link to a file outside the capsule.
    """
    if path.is_symlink() or not path.is_file():
        return False
    ext = os.path.splitext(path.name)[1].lower()
    if ext in _SCAN_EXTS:
        return True
    if ext:
        return False
    try:
        with path.open("rb") as fh:
            return b"\0" not in fh.read(8192)
    except OSError:  # pragma: no cover - read race
        return False


def _answer_targets(
    values: list[object],
) -> tuple[list[tuple[float, int]], list[str]]:
    """Split answers into (numeric round-targets, string targets) for leak matching.

    Numeric: ``(value, decimals)`` for non-integer floats whose canonical form
    has >=4 decimals. A text numeric token is a leak if it ROUNDS to the value
    at that precision — this catches a capsule that stores the answer at FULL
    precision (e.g. ``0.9996090937724982`` for the stated answer ``0.9996``) and
    its x100 percentage rendering. String: distinctive answers (>=8 chars),
    substring-matched. Integers / short / round numbers are skipped so a
    coincidental ``1000`` never triggers a false redaction or build failure.
    """
    nums: list[tuple[float, int]] = []
    strs: list[str] = []
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            try:
                f = float(v)
            except OverflowError as exc:
                raise InvalidReferenceSourceError(
                    "masking numeric reference is unrepresentable"
                ) from exc
            if not math.isfinite(f):
                raise InvalidReferenceSourceError(
                    "masking numeric reference must be finite"
                )
            if f == int(f):  # integers are too generic to match safely
                continue
            dec = len(format(f, "f").rstrip("0").split(".")[-1])
            if dec >= 4:
                nums.append((f, dec))
        else:
            s = str(v).strip()
            if len(s) >= 8:
                strs.append(s)
    return nums, strs


def _token_is_leak(token: str, nums: list[tuple[float, int]]) -> bool:
    """True if numeric ``token`` rounds to any answer (direct or x100 percentage)."""
    try:
        x = float(token)
    except ValueError:  # pragma: no cover - regex guarantees float-parseable
        return False
    for val, dec in nums:
        band = 0.5 * 10 ** (-dec)
        if abs(x - val) <= band or abs(x - val * 100.0) <= band * 100.0:
            return True
    return False


def _redact_text(
    text: str, nums: list[tuple[float, int]], strs: list[str]
) -> tuple[str, bool]:
    """Return (text with every leak occurrence replaced by the marker, changed?)."""
    changed = False
    for s in strs:
        if s in text:
            text = text.replace(s, _VALUE_REDACTION)
            changed = True
    if nums:

        def _sub(m: "re.Match[str]") -> str:
            return _VALUE_REDACTION if _token_is_leak(m.group(0), nums) else m.group(0)

        new = _NUM_TOKEN_RE.sub(_sub, text)
        if new != text:
            changed = True
        text = new
    return text, changed


def _assert_no_value_leak(
    input_dir: Path, values: list[object], *, capsule: str
) -> None:
    """Refuse detected answer matches in supported readable text files.

    The scan uses :data:`_SCAN_EXTS` (plus extensionless text files), a 5 MB limit and existing round-to-answer
    numeric/distinctive-string matching. Oversized, unsupported and unreadable
    files are outside this guard's coverage. A match raises
    :class:`AnswerLeakError`; absence of a match is not proof that the entire
    capsule is free of answer leaks.
    """
    nums, strs = _answer_targets(values)
    if not nums and not strs:
        return
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if not _is_value_scan_target(p):
                continue
            try:
                if p.stat().st_size > 5_000_000:
                    continue
                data = p.read_text(errors="ignore")
            except OSError:  # pragma: no cover - read race
                continue
            for s in strs:
                if s in data:
                    raise AnswerLeakError(
                        f"capsule {capsule}: answer {s!r} still present in {p}"
                    )
            for m in _NUM_TOKEN_RE.finditer(data):
                if _token_is_leak(m.group(0), nums):
                    raise AnswerLeakError(
                        f"capsule {capsule}: answer value ~{m.group(0)} still "
                        f"present in {p} (rounds to a stated answer)"
                    )


def _mask_value_leaks(input_dir: Path, values: list[object]) -> list[str]:
    """Redact answer-value occurrences in shipped text files, in place.

    Companion to :func:`_strip_leak_dirs` / :func:`_strip_leak_files` for the
    stragglers they cannot catch — a value baked into a KEPT file (e.g. an
    extensionless ``code/evaluation`` dump or a result hardcoded in a ``.py``).
    Uses :func:`_redact_text` (the SAME round-to-answer matching as the guard),
    replacing each whole matching numeric token / distinctive string with
    :data:`_VALUE_REDACTION` and leaving the rest of the file as scaffold.
    NOTE: redaction can break a file's syntax when the value sat in code — that
    is acceptable (leak removal is the priority; the operator's ``raw/`` keeps
    the pristine original). Returns the redacted file paths.
    """
    nums, strs = _answer_targets(values)
    if not nums and not strs:
        return []
    masked: list[str] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if not _is_value_scan_target(p):
                continue
            try:
                if p.stat().st_size > 5_000_000:
                    continue
                text = p.read_text(errors="ignore")
            except OSError:  # pragma: no cover - read race
                continue
            new, changed = _redact_text(text, nums, strs)
            if changed:
                p.write_text(new, encoding="utf-8")
                masked.append(str(p))
    return masked


def _strip_notebook_outputs(input_dir: Path) -> list[str]:
    """Clear executed-output cells from every ``*.ipynb`` under ``input_dir``.

    A capsule may ship an EXECUTED notebook (e.g. BixBench's
    ``CapsuleNotebook-*/..._executed.ipynb``) whose code-cell OUTPUTS are the
    very results the task asks the agent to reproduce — an answer leak that
    the whole-dir :func:`_strip_leak_dirs` pass does not catch. This walks
    every notebook under the agent-visible ``input/`` and clears each cell's
    ``outputs`` + ``execution_count`` (nbstripout-style), keeping the notebook
    CODE as a scaffold while removing the computed answers. A no-op for
    capsules without notebooks (CoreBench, BioMysteryBench). The canonical
    answers live only in operator-side ``eval/answers.jsonl`` (never mounted).
    Returns the cleared notebook paths as strings.
    """
    cleared: list[str] = []
    for nb_path in sorted(input_dir.rglob("*.ipynb")):
        if not nb_path.is_file():
            continue
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
        if changed:
            nb_path.write_text(
                json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            cleared.append(str(nb_path))
    return cleared


# EOF
