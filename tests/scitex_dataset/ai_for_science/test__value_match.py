#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/ai_for_science/test__value_match.py

"""Precision-aware answer-value matching (#107) on SYNTHETIC fixtures only.

Every reference value here is generated in-test; no capsule, oracle or
answer file is read. Covers: each rendering class, the explicit
minimum-information rule, redaction + guard sharing ONE matcher, and two
seeded corpora — a leak corpus (recall) and a no-answers corpus
(false-positive rate).
"""

from __future__ import annotations

import random

import pytest

from scitex_dataset.ai_for_science import _leak_strip, _standardize, _value_match
from scitex_dataset.ai_for_science._value_match import (
    CONTEXT_SIG_DIGITS,
    MIN_SIG_DIGITS,
)

REF = 0.873214
MARK = _leak_strip._VALUE_REDACTION


def _redact(text, values=(REF,)):
    nums, strs = _leak_strip._answer_targets(list(values))
    return _leak_strip._redact_text(text, nums, strs)[0]


# ---------------------------------------------------------------------------
# Rendering classes — each must be redacted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "accuracy 87.32%",  # percentage, 2 decimals
        "accuracy 87.3 %",  # percentage, 1 decimal, spaced sign
        "accuracy 87.32 percent",  # x100 without a % sign
        "reached 0.873",  # rounded to 3 significant digits
        "reached 0.8732",  # rounded to 4 significant digits
        "reached 8.73e-01",  # scientific notation
        "reached 8.732E-1",  # scientific notation, upper-case E
        "reached 0,873",  # locale decimal comma
        "reached .873",  # no leading zero
        "reached -0.873",  # negative rendering (abs compared)
        "| run | 0.8732 |",  # Markdown table cell
        "a,b\n1,0.873\n",  # CSV cell
        "reached 0.873214",  # full stored precision
        "reached 0.87321400031",  # MORE precise than stored (raw run output)
        "AUC = 0.87",  # 2 significant digits WITH answer-like context
        "score: 87%",  # 2 significant digits x100 WITH context
    ],
)
def test_rendering_is_redacted(text):
    # Arrange
    # Act
    out = _redact(text)
    # Assert
    assert MARK in out


@pytest.mark.parametrize(
    "text",
    [
        "reached 0.87",  # 2 sig digits, no answer-like context (rule 2)
        "batch 87 items",  # 2 sig digits integer, no context
        "reached 0.874",  # 3 sig digits but outside 0.873214's rounding band
        "reached 0.9",  # 1 sig digit never matches (rule 3)
        "lr = 0.5",  # common number (rule 4)
        "pi ~ 3.14",  # common constant (rule 4)
        "pkg==2.11.0",  # version string is not a token
        "sha a873b21",  # digits inside an identifier are not tokens
    ],
)
def test_non_rendering_is_kept(text):
    # Arrange
    # Act
    out = _redact(text)
    # Assert
    assert out == text


def test_rule_thresholds_are_explicit():
    # Arrange
    # Act
    rule = (MIN_SIG_DIGITS, CONTEXT_SIG_DIGITS)
    # Assert
    assert rule == (3, 2)


def test_integer_reference_is_out_of_scope():
    # Arrange — rule 5: an integer answer never matches (too generic).
    text = "BATCH = 1000\nN = 4237\n"
    # Act
    out = _redact(text, values=(1000, 4237))
    # Assert
    assert out == text


def test_two_digit_reference_needs_context():
    # Arrange — a 2-significant-digit reference carries only 2 digits of
    # information, so even an exact restatement needs answer-like context.
    bare, ctx = "x 0.42 y", "p-value 0.42"
    # Act
    out = (_redact(bare, values=(0.42,)), _redact(ctx, values=(0.42,)))
    # Assert
    assert out == (bare, f"p-value {MARK}")


def test_numeric_string_reference_is_a_numeric_target():
    # Arrange — an oracle string "87.32%" also masks its fraction rendering.
    # Act
    out = _redact("reached 0.8732", values=("87.32%",))
    # Assert
    assert out == f"reached {MARK}"


def test_comma_pair_halves_are_matched_separately():
    # Arrange — "12,873" is not 12.873, but its right half 873 renders
    # 873.214 at integer precision (3 significant digits).
    # Act
    out = _redact("12,873", values=(873.214,))
    # Assert
    assert out == f"12,{MARK}"


def test_guard_refuses_rounded_rendering(tmp_path):
    # Arrange — the backstop uses the SAME matcher: a rounded form is refused.
    (tmp_path / "notes.md").write_text("final accuracy 87.3 %\n")
    # Act
    raised = pytest.raises(_standardize.AnswerLeakError)
    # Assert
    with raised:
        _standardize._assert_no_value_leak(tmp_path, [REF], capsule="c")


def test_guard_message_does_not_carry_the_value(tmp_path):
    # Arrange
    (tmp_path / "notes.md").write_text("final accuracy 87.3 %\n")
    # Act
    try:
        _standardize._assert_no_value_leak(tmp_path, [REF], capsule="c")
        message = ""
    except _standardize.AnswerLeakError as exc:
        message = str(exc)
    # Assert
    assert "answer rendering" in message and "87.3" not in message


def test_mask_then_guard_passes(tmp_path):
    # Arrange — every rendering class in one file; redaction must leave
    # nothing the guard would refuse.
    (tmp_path / "report.txt").write_text(
        "acc 87.32%\nAUC = 0.87\n8.73e-01\n0,873\n| .873 |\n-0.8732\n"
    )
    _standardize._mask_value_leaks(tmp_path, [REF])
    # Act
    _standardize._assert_no_value_leak(tmp_path, [REF], capsule="c")
    # Assert
    assert (tmp_path / "report.txt").read_text().count(MARK) == 6


def test_redaction_reaches_fixpoint_after_string_removal():
    # Arrange — removing the long string answer pulls "score" into the
    # 2-digit token's context window; a second pass must redact it too.
    filler = "x" * 45
    text = f"score {filler} 0.87"
    # Act
    out = _redact(text, values=(REF, filler))
    # Assert
    assert out == f"score {MARK} {MARK}"


# ---------------------------------------------------------------------------
# Seeded synthetic corpora — recall and false-positive rate
# ---------------------------------------------------------------------------


def _random_reference(rng):
    kind = rng.randrange(3)
    if kind == 0:
        return round(rng.uniform(0.05, 0.98), rng.randint(4, 8))
    if kind == 1:
        return round(rng.uniform(1.1, 950.0), rng.randint(3, 6))
    return float(f"{rng.uniform(1.1, 9.9):.4f}e-{rng.randint(3, 6)}")


def _renderings(ref, rng):
    """>=3-significant-digit renderings a capsule file could contain."""
    out = [repr(ref), f"{ref:.3g}", f"{ref:.4g}", f"{ref:.3e}", f"{ref:.2e}"]
    out += [f"-{ref:.4g}", f"| {ref:.4g} |", f"{ref:.3g}".replace(".", ",")]
    if ref < 1:
        out += [f"{ref * 100:.2f}%", f"{ref * 100:.1f} %", f"{ref:.3f}"[1:]]
    return [r for r in out if "e" in r or len(r.strip("-|% ,.0")) >= 3]


def _noise_number(rng):
    kind = rng.randrange(8)
    if kind == 0:
        return str(rng.randint(0, 3000))  # counts, years, line numbers
    if kind == 1:
        return f"{rng.uniform(0, 1):.{rng.randint(1, 6)}f}"  # fractions
    if kind == 2:
        return f"{rng.uniform(0, 100):.{rng.randint(1, 3)}f}%"  # percentages
    if kind == 3:
        return f"{rng.uniform(0, 1000):.{rng.randint(1, 4)}f}"  # magnitudes
    if kind == 4:
        return f"{rng.uniform(1, 9.99):.{rng.randint(1, 3)}f}e-{rng.randint(1, 8)}"
    if kind == 5:
        return rng.choice(["0.5", "0.05", "0.001", "1.0", "100", "3.14159", "1.96"])
    if kind == 6:
        return f"{rng.randint(1, 3)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
    return f"{rng.getrandbits(32):08x}"  # digests / identifiers


_LINES = ("value {} here", "x = {}", "| {} |", "{},{}", "accuracy: {}", "lr {}")


def leak_corpus(seed=7, n_refs=200):
    rng = random.Random(seed)
    cases = []
    for _ in range(n_refs):
        ref = _random_reference(rng)
        for r in _renderings(ref, rng):
            cases.append((ref, f"result line: {r}\n"))
    return cases


def no_answer_corpus(seed=11, n_docs=200, n_tokens=250):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        refs = [_random_reference(rng) for _ in range(rng.randint(1, 3))]
        lines = []
        for _ in range(n_tokens):
            tpl = rng.choice(_LINES)
            nums = [_noise_number(rng) for _ in range(tpl.count("{}"))]
            lines.append(tpl.format(*nums))
        docs.append((refs, "\n".join(lines), n_tokens))
    return docs


def measure():
    """(recall, n_leaks, fp_tokens, n_noise_tokens) on the seeded corpora."""
    hits = 0
    leaks = leak_corpus()
    for ref, text in leaks:
        hits += MARK in _redact(text, values=(ref,))
    fp = total = 0
    for refs, text, _ in no_answer_corpus():
        nums, strs = _leak_strip._answer_targets(refs)
        fp += len(_leak_strip._leak_spans(text, nums, strs))
        total += len(_value_match._NUM_RE.findall(text))
    return hits / len(leaks), len(leaks), fp, total


def test_synthetic_leak_corpus_full_recall():
    # Arrange
    # Act
    recall, n, _, _ = measure()
    # Assert
    assert (recall, n > 1500) == (1.0, True)


def test_synthetic_no_answer_corpus_false_positive_rate_bounded():
    # Arrange
    # Act
    _, _, fp, total = measure()
    # Assert — under 0.5% of independent numeric tokens are redacted.
    assert fp / total < 0.005


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
