# CORE-Bench question and reference identity

The corrected adapter contract is `corebench-question-references-v1`.
One exact question key within one native capsule identifies one assigned
task. Entries of the upstream `results` array are reference runs; their
positions do not create difficulty labels or additional questions.

Task IDs have the form
`corebench/<native-capsule-id>__question_<sha256-of-exact-UTF8-question-key>`.
Question keys are not stripped, case-folded, or otherwise normalised. IDs
therefore survive reordered reference runs and reordered dictionaries.
Changing a reference value does not change the assigned question identity.
Inventory task totals count questions; reference-sample totals are separate.

The solver receives one answer-free task row per question. The private
`answers.jsonl` retains one scalar `{"value": ...}` row per reference
sample, with the **same full task ID** on every sample for that question.
Private metadata records the schema, reference-run index, total number of
runs, reference count, and any runs missing that question. Missing samples
are reported without filling them or silently changing the research policy.
Repeated oracle IDs are reference samples; duplicate submitted IDs are a
different validation problem.

Consumers must read those repeated rows into a list of references per
question. A last-ID-wins dictionary is incompatible with this contract.
The existing Python scorer already groups references by full task ID.
Generated evaluators and downstream callers must explicitly preserve that
cardinality and identify their scoring policy. This adapter change does not
select a new numerical tolerance, aggregation rule, or missingness policy.

Old difficulty-bearing task IDs and retained historical outputs must not
be rewritten to look like corrected results. Use a fresh materialization
for the new contract; existing directories require a matching qualified
private source/cache identity before reuse. The source identity contains
adapter and source-manifest hashes, not reference values in solver files.
Paper-specific loader compatibility and scientific execution are separate
acceptance steps.

`prepare(capsule_ids=[...])` limits capsule acquisition.
`prepare(only=...)` limits capsule materialization while the catalog mapper
remains complete. These are separate scopes; no friendly-ID-to-native-ID
guess is made. Explicit acquisition IDs leave oracle acquisition to the
caller, who must stage the required source manifests before standardizing.
Unsupported `prepare` keywords raise an error instead of being silently
presented as honored.

Synthetic regression examples use invented reference values. They are
format and identity controls, not experimental benchmark results.
