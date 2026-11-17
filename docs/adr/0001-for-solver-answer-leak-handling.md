# ADR 0001 — for_solver answer-leak handling: strip + redact, not just detect

- **Status:** Accepted
- **Date:** 2026-06-23
- **Scope:** `scitex_dataset.ai_for_science` (corebench / bixbench / biomysterybench)

## Status

Accepted.

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

"High-precision" originally meant numeric values with ≥4 significant decimals
(e.g. `0.931818`, `1.469021`) plus their percentage rendering, or ≥8-char
strings. That matching is superseded by the precision-aware matcher of the
2026-11-17 addendum (#107) below. Integers are still **skipped**, so a
coincidental `1000` never triggers a false redaction or build failure.

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
  value-redaction + backstop need the cohort to pass reference values:
  **corebench and bixbench are wired** (bixbench passes `ideal` and `answer`,
  plus the float form of numeric strings, since 2026-11-17).
  **biomysterybench** answers are free-text rubrics, which this exact-value
  policy does not cover; it relies on the structural strips.
- All strips/redactions touch only `for_solver/`; `raw/` is untouched → fully
  reversible and auditable. The materializer returns `stripped_leaks`,
  `stripped_files`, `masked_values`, `cleared_notebooks` for logging.
- Regression coverage in `tests/scitex_dataset/ai_for_science/test__standardize.py`
  (`TestRunningLogLeakStrip`, `TestValueLeakGuard`).
- The value scan also covers EXTENSIONLESS text files (`Dockerfile`, `code/run`,
  `code/evaluation`) and never follows symlinks (2026-11-17). Before this, the
  extension allow-list silently skipped the extensionless dump cited above.

## Addendum (2026-11-17) — hosting identifiers

### Context

After answer masking, prepared CORE-Bench capsules still shipped the
publisher's reproduction notes (`REPRODUCING.md`) naming the capsule's
published image (`registry.codeocean.com/published/<uuid>:vN`), links to the
hosting platform, the Dockerfile's environment digest header, and, in other
capsules, paper/publisher links, DOIs and remote result figures. A solver with
network access can follow any of these to the published results, defeating
value masking. Survey (8 raw CORE-Bench capsules): every capsule had the
published-image reference and hosting links; 6/8 a `# hash:sha256:` header;
one had a publisher URL and four remote result images in its README.

### Decision

A second, always-on pass in the shared writer
(`_identifiers.mask_hosting_identifiers`, then
`assert_no_hosting_identifiers`) rewrites every extracted text file
(sniffed: no NUL byte, any extension, symlinks never followed, read-only
members restored to their mode):

| Class | Rule | Replacement |
|---|---|---|
| Published environment image | `registry.codeocean.com/published/<uuid>[:tag][@sha256:…]` | `capsule-environment:local` (README says how to build it from `input/environment/Dockerfile`) |
| Environment digest | `# hash:sha256:<hex>` line removed; other bare `sha256:<hex>` | `[redacted-digest]` |
| Hosting / publication URL | any URL (scheme optional) on `HOSTING_HOSTS` (capsule hosts, DOI resolvers, archives, preprint servers, publishers) | `[redacted-url]` |
| Result-file link | remote image in Markdown/HTML; URL with a `results/output/figures/plots` segment | `[redacted-url]` |
| DOI | bare `10.NNNN/…` | `[redacted-doi]` |
| Exact identifier | native capsule id (+ UUID core), oracle paper DOI (versioned + unversioned) and title, case-insensitive, whitespace-tolerant | `[redacted-identifier]` |

Kept on purpose: base-image `FROM` lines and `image@sha256:` pins (generic,
needed to rebuild), dependency lists and versions, run commands/scripts,
code-repository and package-index URLs, tool documentation links, contact
emails. The policy version (`IDENTIFIER_POLICY`) and the exact identifiers are
bound into the materialization identity (contract
`per-capsule-source-identity-v2`), so caches built under the old, leakier
policy refuse reuse.

### Limits

- Over-redaction is accepted: a cited third-party DOI or paper link is also
  neutralised; it is never an input needed to rerun the capsule.
- Not removed: author names, affiliations and abstracts (licences and
  attribution) and generic-host links to the authors' own pages. CORE task IDs
  (ADR 0002) still embed the native capsule id. Redaction is therefore defence
  in depth; strict masking also requires a network-isolated solver.
- Text files over 64 MiB are not rewritten; they are listed in the
  materializer's `identifier_unscanned`.
- Regression coverage: `tests/scitex_dataset/ai_for_science/test__identifiers.py`
  over real fixture files in `fixtures/hosting_identifiers/`.

## Addendum (2026-11-17) — precision-aware value matching (#107)

### Context

The value matcher compared text tokens only at the reference's stored
precision. A token matched only if it rounded to the reference at the
reference's decimal count. The same value written to fewer digits slipped past
both the redaction and the fail-loud guard. For a stored `0.873214`, these
forms were missed: `87.32%`, `0.87`, `8.73e-01`, `87.3 %`, `0,873` and a
rounded table cell.

### Decision

There is one matcher, `_value_match.leak_spans`, behind `_mask_value_leaks`
and `_assert_no_value_leak`. Anything the guard refuses is therefore what the
redaction removes. Redaction repeats until no match remains, and the guard
reruns the same matcher on the redacted text.

- **Tokens.** A token is `\d+([.,]\d+)?` or `.\d+`, with an optional `e±N`
  exponent. It cannot follow a letter, digit or dot, and it cannot be
  followed by a digit or `.digit`. As a result, hex digests, identifiers and
  version strings such as `2.11.0` are never tokens. A comma token is read as
  a decimal comma first. If that does not match, each half is matched as an
  integer, which handles CSV cell pairs and thousands separators.
- **Precision-aware comparison.** Each token is parsed with the precision it
  is written at. Its tolerance is half a unit in its last written digit, so
  `87.32` has a tolerance of 0.005 and `8.73e-01` has 0.0005. The token is
  compared with |reference| × {1, 100, 0.01}, ignoring sign. The tolerance
  used is the larger of the token's tolerance and the reference's own scaled
  tolerance. The reference's tolerance applies when the token is more precise
  than the stated value, as with a raw run output.
- **Numeric strings.** A string reference that is itself a number, such as
  `"87.32%"`, is also used as a numeric reference, with the precision it is
  written at.
- **Minimum-information rule (explicit).** Let *agreement* be the smaller of
  the token's and the reference's significant digits.
  1. An agreement of 3 or more significant digits (`MIN_SIG_DIGITS`) always
     counts.
  2. An agreement of exactly 2 (`CONTEXT_SIG_DIGITS`) counts only if an
     answer word appears on the same line within 40 characters before the
     token. Answer words include accuracy, AUC, F1, score, loss, error,
     p-value, mean, median, ratio, percent and result.
  3. One significant digit never counts.
  4. Common numbers never match. These are tokens whose significant digits
     are 1, 2, 5, 15, 25, 75, 125, 375, 625 or 875 at any power of ten,
     written with no more than `max(2, len)` significant digits. Examples are
     0.5, 1.0, 100, 0.25 and 0.125. Values within the token's precision of a
     known constant also never match: pi, e, √2, √3, ln 2, ln 10, 1.96, 1.645,
     2.576, 0.6745 and φ.
  5. Integer references remain out of scope. Integer *tokens* can still
     match a non-integer reference: `873` matches `873.214` because it has 3
     significant digits and a tolerance of 0.5.
- **Identity.** `VALUE_MATCH_POLICY = "precision-aware-values-v1"` is part of
  every capsule's materialization identity, and the contract is now
  `per-capsule-source-identity-v3`. Caches redacted by the older matcher
  refuse reuse.
- **Error text.** `AnswerLeakError` gives the file and the character offset,
  never the matched value.

### Measured on synthetic corpora (no capsule data)

`tests/scitex_dataset/ai_for_science/test__value_match.py` builds two seeded
corpora.

- **Leak corpus.** It has 200 random references (fractions, magnitudes and
  values from 1e-6 to 1e-3), each written in 3 to 11 forms: repr, 3 and 4
  significant digits, 2 and 3 decimals in scientific notation, negative,
  table cell, decimal comma, two kinds of percentage, and no leading zero.
  The matcher redacted all **1747 of 1747** renderings, so recall was 100%.
- **No-answers corpus.** It has 200 documents, each with 1 to 3 random
  references and 250 lines of independent numbers. These include counts,
  fractions, percentages, magnitudes, scientific values, common constants,
  version strings and hex identifiers. The matcher redacted **15 of 48084**
  numeric tokens, a false-positive rate of 0.031%. The test bounds this rate
  below 0.5%.

### Limits

- The rule trades recall for precision only on 2-digit renderings without
  context, such as a bare `0.87` for `0.873214`, and on renderings that
  coincide with a common constant (for example `2.58`, near 2.576).
- A false positive redacts an unrelated number, which can break syntax in
  code. This is the same trade-off already accepted for redaction, and `raw/`
  keeps the original.

## Addendum (2026-11-17) — identifiers in path names (#111)

### Context

Content redaction rewrites file contents only. An upstream archive can also
carry a source identifier in its file and directory NAMES: in at least one
adapter's layout (BixBench) every archive has a top-level folder and notebook
filenames that embed the native capsule UUID or task id. A solver sees those
in any directory listing, and neither the value matcher nor the hosting
matcher looked at paths.

### Decision

The shared writer neutralises path names for every adapter
(`_neutral_paths.neutralize_identifier_paths`), after the structural strips
and BEFORE any content masking, so references are rewritten while the old
names are still literally present:

- **What counts as identifying.** A path component (directory name or file
  name) that contains any UUID, any hit of the content matchers
  (`find_identifiers`: hosting URLs, DOIs, exact identifiers), an exact
  identifier delimited by any non-alphanumeric character (so `_` and `.` act
  as separators), a 6+ digit core of an exact identifier (`capsule-7654321`
  gives `7654321`), or an adapter-supplied id pattern (BixBench passes
  `bix-<n>`).
- **Rename rule.** Each distinct identifying stem gets one neutral stem:
  `notebook_NN` if any entry with that stem is a notebook, else `folder_NN`
  for a directory, else `file_NN`. Numbers are assigned per kind over the
  sorted stems and skip names that already exist. Up to two short
  alphanumeric extensions are kept (`.ipynb`, `.csv`, `.tar.gz`).
- **Reference rewrite.** In every text file (same sniffing as the
  hosting pass, modes preserved) and every symlink target, each old stem is
  replaced by its neutral stem: longest first, case-sensitive, delimited by
  non-alphanumerics. Relative paths in code and notebook cells, Python imports
  of a renamed module, and README links therefore still resolve.
- **Traceability.** The old→new mapping is written to
  `eval/path_renames.json`, keyed by friendly capsule dir. It is never written
  under `for_solver/`, because the old names carry the identifiers.
- **Guard.** `assert_no_identifier_paths` scans every path component of the
  materialized `capsule-NNN/` with the same matchers and raises
  `IdentifierPathLeakError` on a survivor.
- **Notebook structure.** Notebooks are also scanned and redacted
  structurally: every decoded string (keys and values) of notebook-level
  metadata, cell metadata, sources, outputs and attachments. A JSON escape
  such as `10.5555\/x` can no longer hide an identifier from the raw-text
  scan. Outputs are still cleared by the answer pass.
- **Versioning.** `IDENTIFIER_POLICY` is `hosting-identifiers-v2`, the path
  policy `neutral-paths-v1` and the adapter path patterns are bound into each
  capsule's identity, and the contract is `per-capsule-source-identity-v4`.
  Outputs prepared under v3 are refused; prepare them again into a fresh
  directory.

### Limits

- A path built at run time from fragments (for example
  `f"{uuid}.csv"` with the UUID held in a variable) is not rewritten, because
  the full old name never appears literally. The content pass still redacts
  the identifier itself.
- An identifier stem that also occurs in prose is rewritten to the neutral
  name instead of a `[redacted-*]` marker. It is still neutral.
- Regression coverage: `tests/scitex_dataset/ai_for_science/test__neutral_paths.py`
  over synthetic archives built in the test (no benchmark data). The tests
  run the renamed script, module import and notebook cells to prove the
  references still resolve.
