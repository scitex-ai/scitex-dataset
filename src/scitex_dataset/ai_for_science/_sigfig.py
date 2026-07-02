#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_sigfig.py

"""Sig-fig-aware float comparison (ported from paper-scitex-clew).

A small, reusable comparator that compares two floats at the
significant-figure precision of the *reference* value. Used by the
numeric scoring family (:mod:`._score`) as the ``n == 1`` /
degenerate-prediction-interval fallback: when the oracle carries a
single reference value (or all reference reruns agree exactly), an
exact-equality check demands more precision than the published number
claims, so we compare at the reference's stated sig-fig precision
instead.

This is a verbatim port of
``scripts/cohorts/_shared/verify/_sigfig.py`` at revision
``2bc727526`` in paper-scitex-clew, so the host-side scorer and the
clew verifier agree bit-for-bit.

- :func:`sigfigs` — count significant figures of a number from its
  shortest-unique Python ``str`` repr. Examples::

    sigfigs("0.9996")            -> 4
    sigfigs("0.59375")           -> 5
    sigfigs(0.6733021077)        -> 10
    sigfigs(0.9990000128746033)  -> 16
    sigfigs("1.00e3")            -> 3
    sigfigs(0)                   -> 1

- :func:`is_close_sigfig` — ``math.isclose``-style comparator whose
  ``rel_tol`` is derived from the reference value's sig figs:
  ``rel_tol = factor * 10 ** (-sigfigs(b))``. Default ``factor=5``
  gives half-an-ULP at the reference's least significant digit.
  ``abs_tol=1e-9`` handles the ``b == 0`` degenerate case.
"""

from __future__ import annotations

import math


def sigfigs(x) -> int:
    """Return the count of significant figures of ``x``.

    ``str`` inputs are taken at face value (the caller asserts the
    precision of the published number). ``int`` / ``float`` inputs are
    first converted to their shortest-unique Python repr.

    Edge cases:

    - Negative values: sign is stripped before counting.
    - Scientific notation: only the mantissa contributes.
    - Pure-integer with trailing zeros (e.g. ``"100"``): trailing
      zeros are ambiguous; conservatively counted as 1 SF. Use
      ``"1.00e2"`` to disambiguate.
    - Zero: 1 SF.
    """
    s = str(x).strip()
    if not s:
        return 0
    if s[0] in ("+", "-"):
        s = s[1:]
    if not s:
        return 0
    for sep in ("e", "E"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    if float(s) == 0:
        return 1
    if "." in s:
        before, after = s.split(".", 1)
        before = before.lstrip("0")
        if before:
            return len(before + after)
        return len(after.lstrip("0"))
    return len(s.lstrip("0").rstrip("0")) or 1


def is_close_sigfig(a, b, factor: float = 5.0, abs_tol: float = 1e-9) -> bool:
    """``math.isclose`` with ``rel_tol = factor * 10 ** (-sigfigs(b))``.

    Parameters
    ----------
    a, b : float or str
        ``b`` is the reference (oracle) value; ``a`` is the reported
        (agent) value. ``b`` may be passed as a string to assert a
        specific sig-fig count.
    factor : float
        Multiplier on the half-ULP tolerance. Default 5.0 gives
        half-an-ULP at the reference's least significant digit.
    abs_tol : float
        Fallback absolute tolerance, used when ``b == 0``.

    Returns
    -------
    bool
        True iff ``a`` and ``b`` agree at the reference's published
        precision.
    """
    n = sigfigs(b)
    rel = factor * (10 ** (-n))
    a_f = float(a)
    b_f = float(b)
    return math.isclose(a_f, b_f, rel_tol=rel, abs_tol=abs_tol)


__all__ = ["sigfigs", "is_close_sigfig"]

# EOF
