#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_value_match.py

"""Precision-aware numeric answer matching (ADR 0001, addendum #107).

ONE matcher shared by the in-place redaction (:func:`._leak_strip._mask_value_leaks`)
and the fail-loud backstop (:func:`._leak_strip._assert_no_value_leak`), so
whatever the guard would refuse is exactly what the redaction removes.

Why precision-aware: a capsule file rarely restates a reference value at its
stored precision. A stored ``0.873214`` shows up as ``87.32%``, ``0.873``,
``8.73e-01``, ``87.3 %``, ``0,873``, ``.873`` or a table cell. Each numeric
TOKEN in the text is therefore parsed together with its own WRITTEN precision
and compared against every reference value (and its x100 / x0.01 percentage
renderings, sign ignored) within the rounding tolerance that precision
implies::

    token "87.32"  -> value 87.32, last written digit 1e-2 -> tolerance 0.005
    0.873214 x 100 = 87.3214;  |87.32 - 87.3214| = 0.0014 <= 0.005  -> MATCH

A token written MORE precisely than the reference (an un-rounded run output
``0.99960909...`` for a stated ``0.9996``) matches within the reference's own
rounding tolerance instead.

Minimum-information rule (explicit; see :data:`MIN_SIG_DIGITS`)
---------------------------------------------------------------
A coincidental agreement on one or two digits is not evidence of a leak, so a
match also needs the AGREEMENT — ``min(token sig. digits, reference sig.
digits)`` — to carry enough information:

1. ``>= 3`` significant digits (:data:`MIN_SIG_DIGITS`) always qualifies;
2. exactly ``2`` (:data:`CONTEXT_SIG_DIGITS`) qualifies ONLY when an
   answer-like keyword (:data:`_CONTEXT_RE` — accuracy, AUC, score, p-value,
   mean, ...) occurs on the same line within :data:`CONTEXT_WINDOW` characters
   before the token (``"AUC = 0.87"`` matches, a bare ``0.87`` does not);
3. ``1`` significant digit never qualifies;
4. trivially common numbers never match, whatever their context: a token
   whose significant digits are one of :data:`_COMMON_MANTISSAS` (1, 2, 5,
   10, 100, 0.5, 0.25, 0.75, 0.125, 1.5, 1.0, 0.50, ...) written with no
   more than ``max(2, len(mantissa))`` significant digits (so ``2.50e-04`` or
   ``0.250`` — which state extra precision — stay matchable), or that equals a
   well-known constant (:data:`_COMMON_CONSTANTS` — pi, e, z-scores, sqrt 2,
   ln 2, ...) within its written precision;
5. INTEGER references are out of scope (as before): an integer count is too
   generic to match safely. Integer TOKENS still match a non-integer
   reference (``873`` for ``873.214``) under rules 1-4, with tolerance 0.5.

Token grammar: ``\\d+([.,]\\d+)?`` or ``.\\d+``, optional ``e±N`` exponent; not
preceded by a letter/digit/dot (so hex digests, identifiers and version
strings such as ``2.11.0`` are never tokens) and not followed by a further
digit or ``.digit``. A comma token (``0,873``) is read as a locale decimal
comma first; when that does not match, its two halves are matched as separate
integers (a CSV ``0,873`` cell pair or a thousands separator).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Version of the matching policy — bound into the materialization identity so a
# cache built under an older (leakier) policy refuses reuse.
VALUE_MATCH_POLICY = "precision-aware-values-v1"

MIN_SIG_DIGITS = 3
CONTEXT_SIG_DIGITS = 2
CONTEXT_WINDOW = 40

# Renderings compared: as stored, as a percentage, and a stored percentage
# rendered as a fraction. The sign is ignored (|token| vs |reference|).
_SCALES = (1.0, 100.0, 0.01)
_REL_EPS = 1e-9

_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<mant>\d+(?:[.,]\d+)?|\.\d+)"
    r"(?P<exp>[eE][-+]?\d+)?"
    r"(?!\.?\d)"
)
# Significant-digit strings of trivially common numbers (rule 4): 1, 2, 5 and
# 2.5 / 7.5 / 1.25 / ... times any power of ten.
_COMMON_MANTISSAS = frozenset(
    {"1", "2", "5", "25", "75", "125", "375", "625", "875", "15"}
)
_COMMON_CONSTANTS = (
    math.pi,
    math.e,
    math.sqrt(2.0),
    math.sqrt(3.0),
    math.log(2.0),
    math.log(10.0),
    1.96,
    1.645,
    2.576,
    0.6745,
    1.6180339887,
)
_CONTEXT_RE = re.compile(
    r"(accura|\bacc\b|\bauc|auroc|auprc|\broc\b|\bf1\b|precision|recall|"
    r"sensitivity|specificity|score|\br2\b|r\^2|r²|rmse|\bmse\b|\bmae\b|loss|"
    r"error|correlat|\bcorr\b|pearson|spearman|p[-_ ]?val|\bp\s*[=<>]|"
    r"odds|hazard|\bmean\b|median|average|answer|result|estimate|coef|"
    r"\bbeta\b|ratio|rate\b|fold|log2fc|percent|proportion|fraction|"
    r"value\b|\bstd\b|variance|\bci\b|interval)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericTarget:
    """A reference value with its own written precision."""

    value: float  # absolute value
    sig: int  # significant digits of its shortest round-trip rendering
    tol: float  # half a unit in its last written digit


@dataclass(frozen=True)
class _Token:
    value: float  # absolute value
    sig: int
    tol: float
    digits: str  # significant-digit string (for the common-number rule)


def _parse(mant: str, exp: str | None) -> _Token | None:
    """Parse a numeric token into value + written precision; None if empty."""
    has_point = "." in mant or "," in mant
    int_part, _, frac = mant.replace(",", ".").partition(".")
    digits = (int_part + frac).lstrip("0")
    if has_point:
        sig_digits = digits  # trailing zeros after the point are significant
    else:
        sig_digits = digits.rstrip("0")
    if not sig_digits:
        return None
    e = int(exp[1:]) if exp else 0
    try:
        value = abs(float(f"{int_part or '0'}.{frac or '0'}e{e}"))
    except OverflowError:  # pragma: no cover - absurd exponent
        return None
    if not math.isfinite(value):
        return None
    tol = 0.5 * 10.0 ** (e - len(frac))
    return _Token(value, len(sig_digits), tol, digits.rstrip("0"))


def _target(tok: _Token | None) -> NumericTarget | None:
    if tok is None or tok.value == int(tok.value):
        return None  # rule 5: integer (and zero) references are out of scope
    return NumericTarget(tok.value, tok.sig, tok.tol)


def numeric_target(value: float) -> NumericTarget | None:
    """Target for a finite NON-integer reference; None for integers / zero.

    Precision is that of the shortest round-trip rendering (``repr``).
    """
    mant, _, exp = repr(abs(float(value))).partition("e")
    return _target(_parse(mant, "e" + exp if exp else None))


def numeric_string_target(text: str) -> NumericTarget | None:
    """A numeric STRING reference (``"0.8732"``, ``"-87.32 %"``) as a target.

    Precision is the string's own written precision; None if not a number.
    """
    s = text.strip()
    s = s[:-1].rstrip() if s.endswith("%") else s
    m = _NUM_RE.fullmatch(s.lstrip("-+\u2212"))
    return _target(_parse(m.group("mant"), m.group("exp"))) if m else None


def _is_common(tok: _Token) -> bool:
    """Rule 4: a trivially common number or a well-known constant."""
    # "1.0" / "0.50" / "0.125" are common; "2.50e-04" / "0.250" state more
    # precision than the common number itself and stay matchable.
    if tok.digits in _COMMON_MANTISSAS and tok.sig <= max(2, len(tok.digits)):
        return True
    return any(abs(tok.value - c) <= tok.tol for c in _COMMON_CONSTANTS)


def _has_context(text: str, start: int) -> bool:
    """Rule 2: an answer-like keyword on the same line just before ``start``."""
    window = text[max(0, start - CONTEXT_WINDOW) : start]
    return _CONTEXT_RE.search(window.rsplit("\n", 1)[-1]) is not None


def _token_matches(
    tok: _Token | None, targets: list[NumericTarget], text: str, start: int
) -> bool:
    """True if ``tok`` renders a target within ITS OWN rounding tolerance."""
    if tok is None or tok.sig < CONTEXT_SIG_DIGITS or _is_common(tok):
        return False
    context: bool | None = None
    for t in targets:
        agreement = min(tok.sig, t.sig)
        if agreement < CONTEXT_SIG_DIGITS:
            continue
        if not any(
            abs(tok.value - t.value * scale)
            <= max(tok.tol, t.tol * scale) + _REL_EPS * t.value * scale
            for scale in _SCALES
        ):
            continue
        if agreement >= MIN_SIG_DIGITS:
            return True
        if context is None:
            context = _has_context(text, start)
        if context:
            return True
    return False


def leak_spans(text: str, targets: list[NumericTarget]) -> list[tuple[int, int]]:
    """Character spans of every numeric token in ``text`` matching a target.

    The single matcher behind both redaction and the fail-loud guard.
    """
    spans: list[tuple[int, int]] = []
    if not targets:
        return spans
    for m in _NUM_RE.finditer(text):
        mant, exp = m.group("mant"), m.group("exp")
        if _token_matches(_parse(mant, exp), targets, text, m.start()):
            spans.append(m.span())
        elif "," in mant and not exp:
            # Not a decimal-comma rendering: match each half as an integer
            # (a CSV ``0,873`` cell pair or a thousands separator).
            left, right = mant.split(",", 1)
            for part, begin in ((left, m.start()), (right, m.start() + len(left) + 1)):
                if _token_matches(_parse(part, None), targets, text, begin):
                    spans.append((begin, begin + len(part)))
    return spans


__all__ = [
    "VALUE_MATCH_POLICY",
    "MIN_SIG_DIGITS",
    "CONTEXT_SIG_DIGITS",
    "CONTEXT_WINDOW",
    "NumericTarget",
    "numeric_target",
    "numeric_string_target",
    "leak_spans",
]

# EOF
