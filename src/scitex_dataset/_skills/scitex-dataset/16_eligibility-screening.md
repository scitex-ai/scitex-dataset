---
description: |
  [TOPIC] Eligibility screening of benchmark units
  [DETAILS] Choose which CORE-Bench / BixBench / BioMysteryBench units qualify for an experiment from a declarative YAML config; every unit gets a verdict and a reason.
tags: [scitex-dataset-eligibility-screening, scitex-dataset-ai-for-science, scitex-dataset]
---

# Eligibility screening

Which capsules or problems qualify for an experiment is a config, not code.
Adapters supply facts for each unit and one engine applies the rules the same
way for every benchmark. The result lists every unit in a fixed order, so a
selection can be frozen and audited.

```bash
scitex-dataset ai-for-science corebench screen \
  --config criteria.yaml --raw-dir /data/ai-for-science/corebench/raw \
  --fetch --out eligibility.json
```

```python
from scitex_dataset.ai_for_science import screen_eligibility, write_eligibility_result
result = screen_eligibility("criteria.yaml", raw_dir=raw, fetch=True)
write_eligibility_result(result, "eligibility.json")
result["selected"]    # native ids in order; pass each to prepare(only=...)
```

## Config (`scitex-dataset/eligibility-config-v1`)

```yaml
benchmark: corebench            # corebench | bixbench | biomysterybench
source: {revision: <hf-sha>}    # bixbench/biomysterybench pin; variant: preview|full
rules:                          # all must hold
  - {id: split, fact: split, in: [train]}
  - {id: size, fact: archive_bytes, max: 1000000000}
  - {id: cpu_only, fact: gpu_in_environment, equals: false}
  - id: light_deps
    all:
      - {fact: dl_frameworks, none_of: [tensorflow, torch, keras, jax]}
      - {fact: code_languages, none_of: [r]}
order:
  keys: [{fact: n_questions, descending: true}, {fact: unit_id}]
  seed: null                    # an int adds a seeded random tie-break
target: 4                       # the first N survivors; null screens all
on_unknown: exclude             # a rule on an unknown fact: exclude | error
```

Scalar operators are `equals`, `in`, `not_in`, `min` and `max`. List
operators are `any_of`, `none_of` and `subset_of`. Use `all` or `any` to
combine predicates. `extra.<key>` reads an adapter's own metadata, for
example BioMysteryBench `extra.human_solvable`.

## Facts

Metadata facts are cheap and come from the benchmark manifest:

- `unit_id`, `label`, `split`, `declared_language`
- `n_questions`, `n_figure_questions`, `archive_bytes`

CORE-Bench uses the upstream `fig` key marker to count figure questions; the
other benchmarks use a keyword screen of the question text. Only question
text, keys and descriptive fields are read; reference values never are.

Archive facts come from the unit's archive and use the same analysis for every
benchmark:

- languages: `file_languages`, `kernel_languages`, `magic_languages`,
  `toolchain_languages`, and their union `code_languages`
- `dl_frameworks`
- GPU: `gpu_in_environment`, `gpu_in_code`, and their union `gpu_required`
- entry: `entry_kinds`, `n_entry_steps`, `entry_notebook_code_cells`,
  `entry_notebook_local_imports`

CORE-Bench's entry is `code/run` (or `run.sh`). BixBench and BioMysteryBench
ship no run script, so every notebook and script counts as an entry.
`facts.evidence` shows the matched lines.

## Semantics

1. Metadata rules are checked for every unit in declared order. The first
   failing rule excludes the unit, and no download is made for it.
2. The remaining candidates are sorted by `order.keys`, then by the seeded
   draw, then by natural `unit_id`. Nulls sort last.
3. The engine then walks the candidates in that order. It reads each
   candidate's archive (fetching it if `fetch` is on) and evaluates every
   archive rule. Each candidate is `selected` or `excluded`. An excluded
   candidate lists all the rules it failed.
4. After `target` units are selected, the rest are `not_screened`. If the
   config has no archive rules they are `eligible_not_needed` instead.

A needed archive that is not staged raises `ArchiveUnavailableError` when
`fetch` is off; it is never skipped. Eligibility does not certify that a
unit's inputs are free of answer leaks. `prepare` and `build_solver_project`
still apply their masking guards.
