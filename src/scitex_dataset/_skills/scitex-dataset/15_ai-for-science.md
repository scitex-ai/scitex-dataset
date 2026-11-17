---
description: |
  [TOPIC] AI-for-science benchmark workflow
  [DETAILS] Prepare private evaluator references and isolated solver assignments; validate and score benchmark submissions.
tags: [scitex-dataset-ai-for-science, scitex-dataset]
---

# AI-for-science benchmark workflow

Use `scitex_dataset.ai_for_science` for CORE-Bench, BixBench and
BioMysteryBench. Keep upstream `raw/` and host-side `eval/` private. Bind only
the selected `for_solver/capsule-NNN/` directory to a solver. The catalog
index and operator materialization ledger are not a selected assignment.
Preparation applies existing masking rules; it does not prove that a real
capsule is free of answer leaks. Qualify the selected input before experiments.

Masking has two layers (ADR 0001). Answer masking strips author output/log
dirs and files, clears notebook outputs, and redacts reference values (CORE
and BixBench pass their oracle values). Hosting-identifier redaction then
rewrites every extracted text file: published environment images become the
local tag `capsule-environment:local` (build it from
`input/environment/Dockerfile`), Dockerfile `# hash:sha256:` headers and bare
digests, capsule-hosting/archive/publisher URLs, remote result images and
`results/`-path links, DOIs, the native capsule id and the oracle's paper
title/DOI are replaced by `[redacted-*]` markers. Dependency lists, package
versions, base-image `FROM` lines, run commands and code-repository URLs stay.
A surviving identifier raises `HostingIdentifierLeakError`; files over 64 MiB
are reported as `identifier_unscanned`. Author names and abstracts remain and
CORE task IDs still embed the native capsule id, so a solver that must not
look up published results also needs a network-isolated sandbox.

Reference-value redaction is precision-aware (`_value_match`,
`precision-aware-values-v1`). Each numeric token in a text file is compared
with every non-integer reference, its x100 and x0.01 forms, sign ignored,
within the token's own rounding tolerance. Stored `0.873214` therefore also
matches `87.32%`, `87.3 %`, `0.873`, `8.73e-01`, `0,873`, `.873`, `-0.873` and
table or CSV cells. Minimum-information rule: the agreement needs at least 3
significant digits. With 2 it needs an answer word (accuracy, AUC, score,
p-value, mean, ...) before the token on the same line, so `AUC = 0.87`
matches but a bare `0.87` does not. One digit, common numbers such as 0.5, 1.0
and 0.125, constants such as pi and 1.96, and integer references never match.
Redaction and the `AnswerLeakError` guard share this matcher. Caches made by
the older matcher are refused (contract `per-capsule-source-identity-v3`).

CORE downloads count a capsule as fetched only after verification. The body
length must equal `Content-Length`; the file is written as `.part` and then
renamed. The `.tar.gz` must read to the end, and any published sha256 passed
as `expected_sha256={id: hex}` must match. A failed check is retried 3 times.
After that the capsule is counted in `n_failed` and leaves no file. `prepare`
raises `IncompleteDownloadError`, and CLI `download` exits 1. Existing files
that are not in `.checksums.json` (hand-staged or partial) are checked as
archives before they are skipped.

Prepare fresh outputs when upgrading the assignment contract or changing
source bytes. CORE `prepare(capsule_ids=[...])` selects native capsules for
acquisition; `only=...` selects materialization and leaves the catalog mapper
complete. These scopes differ. For HuggingFace adapters, `revision` selects
the requested revision; the returned descriptor is not a resolved immutable
commit. BioMysteryBench full uses separate raw and output paths from preview;
consume the paths returned by `prepare(download_full=True)`. The CLI passes
`--full` only to adapters that accept it. CORE has no preview split, so
`corebench prepare --full` prepares the default full set.

CORE assigns one task per exact question key, using
`corebench/<native-id>__question_<full-sha256>`. Private oracle rows repeat
that same ID for reference-run samples. Group them into a list per
question. Do not derive difficulty from reference position, overwrite
repeated rows, or migrate old difficulty-bearing solver results.

Validate with the selected public task file:

```bash
scitex-dataset ai-for-science corebench validate \
  --submission submission.json --tasks /selected/capsule/task.jsonl --json
```

Submissions are JSON arrays of objects with `task_id` and `answer`. Submit
each assigned task exactly once. Use `answer: null` with a nonempty string
`reason` to abstain. Optional public `answer_type` values are `json`, `number`,
`integer`, `string`, `boolean`, `list` and `object`. Validation rejects
coercion and nonfinite numbers, reports public field/type errors, and never
reads references or grades correctness. An absent declaration permits
supported JSON; it does not establish a scientific numeric answer type.
Declare expected types publicly before an experiment, never infer them from
private references. Unknown JSON fields remain warnings.

Use the same public assignment and validation feedback in every comparison
arm. Experiment retry limits are an experiment design decision; validation
does not choose them. Missing metadata means only shape validation, not
selected task/type acceptance. The gate refuses ambiguous sibling capsules.

Host `score_submission` and generated `eval/evaluate.py` have distinct
policies. The host retains its pooled-reference prediction interval and
significant-figure fallback. Generated scalar numeric grading retains its
relative tolerance; multiple numeric/string references report
`needs_reference_policy` until an aggregation policy is specified. Invalid
references are evaluator problems, not solver failures. Missing or malformed
answers with gradeable references remain failures in the generated score.
Inspect total, scored and ungradeable counts; `score: null` means no gradeable
score was defined. The host API defaults to `strict_oracle=True` and refuses
malformed oracle records; explicitly disabling it is diagnostic mode, not
oracle-integrity qualification. Source tests are not scientific benchmark
results.
