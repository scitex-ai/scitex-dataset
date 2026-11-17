#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_reference.py

"""Reference-payload handling for the host-side scorer (split from ``_score``).

Each adapter writes ``eval/answers.jsonl`` records as
``{"task_id", "answer": <payload>}`` and declares, as ``REFERENCE_FIELD``,
which key of ``<payload>`` holds the gradeable reference:

- corebench ``{"value": v}`` -> ``value``
- bixbench ``{"answer": <bool hypothesis flag>, "ideal": <str>}`` -> ``ideal``
- biomysterybench ``{"rubric": r}`` -> ``rubric``

The scorer reads exactly that key. Guessing among candidate keys is what
graded every BixBench task against the boolean ``answer`` flag (#115).
"""

from __future__ import annotations

import math
import re
from typing import Any


def reference_fields() -> dict[str, str]:
    """``{benchmark: payload key}`` from each adapter's ``REFERENCE_FIELD``."""
    from . import biomysterybench, bixbench, corebench

    return {
        m.BENCHMARK: m.REFERENCE_FIELD for m in (corebench, bixbench, biomysterybench)
    }


def is_blank_reference(value: Any) -> bool:
    """True for a null or whitespace-only text reference (ungradeable)."""
    return value is None or (isinstance(value, str) and not value.strip())


def reference_issue(payload: Any, reference_field: str | None) -> str | None:
    """Diagnostic kind when ``payload`` lacks a gradeable reference, else None.

    Only dict payloads are checked (a bare scalar / null payload keeps its
    legacy per-task handling). Kinds: ``missing_reference_field`` /
    ``empty_reference``. Never returns the reference value itself.
    """
    if reference_field is None or not isinstance(payload, dict):
        return None
    if reference_field not in payload:
        return "missing_reference_field"
    if is_blank_reference(payload[reference_field]):
        return "empty_reference"
    return None


# ``(lo,hi)`` / ``[lo, hi]`` interval references (BixBench range verifier).
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_RE = re.compile(rf"^\s*[\(\[]\s*({_NUM})\s*,\s*({_NUM})\s*[\)\]]\s*$")


def parse_interval(value: Any) -> tuple[float, float] | None:
    """``"(lo,hi)"`` -> ``(lo, hi)`` ordered and finite; anything else -> None."""
    if not isinstance(value, str):
        return None
    m = _INTERVAL_RE.match(value)
    if m is None:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (min(lo, hi), max(lo, hi))


def expected_value(payload: Any, reference_field: str | None = None) -> Any:
    """Pull the comparable value out of an answers.jsonl ``answer`` payload.

    With the adapter-declared ``reference_field`` the value is read from
    exactly that key (the oracle loader has already refused payloads where
    it is missing or blank). Without one (an unregistered benchmark) the
    legacy first-non-null of ``value`` / ``answer`` / ``ideal`` applies and
    any other dict / scalar is returned as-is. Rubric payloads are handled
    by the scorer's rubric routing.
    """
    if isinstance(payload, dict) and reference_field is not None:
        return payload.get(reference_field)
    if isinstance(payload, dict):
        for key in ("value", "answer", "ideal"):
            if key in payload and payload[key] is not None:
                return payload[key]
        return payload
    return payload


__all__ = [
    "reference_fields",
    "is_blank_reference",
    "reference_issue",
    "parse_interval",
    "expected_value",
]

# EOF
