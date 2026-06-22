# ADR 0001 — for_solver answer-leak handling: strip + redact, not just detect

- **Status:** Accepted
- **Date:** 2026-06-23
- **Scope:** `scitex_dataset.ai_for_science` (corebench / bixbench / biomysterybench)

## Context

CoreBench and similar *reproducibility* benchmarks ship the original authors'
**run outputs** inside each capsule — result dumps, training/eval logs — and
those contain the very metric the task asks the agent to *reproduce*. An audit
of all 90 raw CoreBench capsules found **42/88 leak** at least one answer value
through a shipped output file.

`for_solver/` is the agent-visible view; the answer must never be reachable
there. The operator's `raw/` snapshot is private and retains the pristine
capsule, so any strip/redaction on `for_solver/` is non-destructive and
reversible.

The pre-existing leak-strip removed only a **top-level `results/`** directory.
It missed:

1. nested output dirs — `code/dump/`, `code/log/` (e.g. capsule-0220918);
2. loose log files outside leak dirs — `data/*_log.csv` (capsule-1900704);
3. values baked into an otherwise-needed, **kept** file — an extensionless
   `code/evaluation` dump (capsule-1900704).

## Decision

Defense in depth in the **shared** `write_for_solver_per_capsule` (so one fix
covers every per-capsule cohort):

1. **Structural dir strip** (`_strip_leak_dirs`) — remove
   `results / result / output(s) / log(s) / dump(s)` dirs at **any depth**
   (was top-level `results/` only).
2. **Loose log-file strip** (`_strip_leak_files`) — remove `*.log` plus files
   whose stem has a **delimited** `log` segment, restricted to text/data
   extensions (so `catalog.csv`, `log_utils.py` are kept).
3. **Value redaction** (`_mask_value_leaks`) — for stragglers (2)+(3) the
   structural passes cannot catch, replace every **high-precision** rendering
   of the capsule's answers with `[redacted]`, in place, boundary-aware,
   longest-form-first. The file is otherwise **kept** as scaffold.
4. **Fail-loud backstop** (`_assert_no_value_leak`) — re-scan after the above;
   if any high-precision value still appears, raise `AnswerLeakError`.

"High-precision" = numeric values with ≥4 significant decimals (e.g.
`0.931818`, `1.469021`) plus their percentage rendering, or ≥8-char strings.
Integers and short/round numbers are **skipped**, so a coincidental `1000`
never triggers a false redaction or build failure.

## Why redact in place (not remove the whole file)

Removing any straggler file wholesale was considered (an output file is
disposable). We chose **in-place redaction** to *preserve the file as
scaffold* — code structure, schema, surrounding prose — so the agent can still
see *how* to reproduce without seeing *what* the answer is. Accepted
trade-offs:

- Masking must catch every rendering — mitigated by enumerating the decimal +
  percentage forms and by the fail-loud backstop (a missed rendering fails the
  build rather than shipping a leak).
- Context is preserved, so the agent may infer that *a* value existed
  (`final accuracy = [redacted]`). Intentional: it signals the quantity to
  reproduce without revealing it.
- Redaction can break a file's syntax when the value sat in code. Acceptable —
  leak removal dominates and `raw/` keeps the pristine original.

## Consequences

- corebench/bixbench/biomysterybench all route through the shared writer, so
  the structural strips + notebook-output scrub apply to every cohort. The
  value-redaction + backstop need the cohort to pass `answer_values`: **corebench
  is wired; bixbench/biomysterybench are a follow-up.**
- All strips/redactions touch only `for_solver/`; `raw/` is untouched → fully
  reversible and auditable. The materializer returns `stripped_leaks`,
  `stripped_files`, `masked_values`, `cleared_notebooks` for logging.
- Regression coverage in `tests/scitex_dataset/ai_for_science/test__standardize.py`
  (`TestRunningLogLeakStrip`, `TestValueLeakGuard`).
