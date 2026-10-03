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

Prepare fresh outputs when upgrading the assignment contract or changing
source bytes. CORE `prepare(capsule_ids=[...])` selects native capsules for
acquisition; `only=...` selects materialization and leaves the catalog mapper
complete. These scopes differ. For HuggingFace adapters, `revision` selects
the requested revision; the returned descriptor is not a resolved immutable
commit. BioMysteryBench full uses separate raw and output paths from preview;
consume the paths returned by `prepare(download_full=True)`.

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
