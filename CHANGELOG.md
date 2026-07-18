# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-07-19

### Changed
- `validate_submission` now enforces the honest-abstention half of the
  submission contract: an entry whose `answer` is `null` MUST carry a
  non-empty `reason`. A missing `reason` key, `reason: null`, `""`, or a
  whitespace-only reason yields a hard `missing_reason` finding naming the
  offending `task_id` (`ok=False`); previously such reasonless nulls were
  silently accepted. Answered (non-null) claims are unaffected — `reason`
  stays optional there. The `dataset-submission-format` pre-submission gate
  inherits the rule (with a `missing_reason` fix-hint), closing the gap
  where the gate accepted silent no-answers.

## [0.7.0] - 2026-07-03

### Added
- Submission-gate plugin provider for `scitex-dev gate` — registers a
  `pre-submission` `GateCheck` (`id="dataset-submission-format"`) under the
  `scitex_dev.gate.checks` entry-point group. It locates the bound capsule's
  submission file (`submission/submission.json` by default; overridable via
  the gate config's `submission_file`) and runs the oracle-free
  `validate_submission`, mapping the result to a `GateResult` +
  format-specific `Finding` fix-hints. benchmark + expected task_ids are
  read from the capsule's own `task.jsonl` (per-capsule), with a
  structure-only fallback when absent. The check is fail-closed and the
  plugin shim defers its `scitex_dev.gate` import, so importing
  `scitex_dataset` never requires scitex-dev to be installed.

## [0.6.0] - 2026-07-03

### Added
- `score_submission()` API + `score` CLI verb — host-side submission grading
  primitive with a 5-way verdict (correct / wrong / abstain / malformed /
  needs_rubric) and numeric (Student-t 95% prediction interval with sig-fig
  fallback), string, and order-insensitive set-equality evaluators.
- `validate_submission()` API + `validate` CLI verb — oracle-free structural
  schema conformance for a submission, pairing with the scorer.
- CORE-Bench `download --full` oracle bootstrap — auto-fetch + GPG-decrypt of
  `core_train` / `core_test` into the operator-private `raw/` (never mounted),
  with a sha256 checksum ledger for resumable re-runs.
- Per-capsule **source registration** — host-side `eval/sources.jsonl` marking
  each capsule's legitimate sources (raw problem data/code + the computed
  output of running the analysis, captured from the pristine raw archive) and
  excluding README / REPRODUCING / paper docs, closing the README
  score-table grounding leak.

### Changed
- `for_solver/` standardized into the per-capsule contract — one
  self-contained `capsule-NNN/` dir (extracted `input/`, `task.jsonl`, uniform
  submission schema/example, README) plus a root `index.jsonl` mapper
  (`friendly_id` ↔ `native_id`) — across CORE-Bench, BixBench, and
  BioMysteryBench.

### Fixed
- BixBench `task_id` now keys on the unique `question_id` (all 205 questions)
  instead of the capsule `short_id`, which collapsed them to 54.
- CI test failures on `develop` from the incomplete per-capsule migration.

## [0.3.1] - 2026-03-29

### Fixed
- Version consistency: aligned __version__, pyproject.toml, and sphinx conf to match
- Fixed __version__ showing 0.2.1 instead of correct version in CLI output

## [0.3.0] - 2026-03-29

### Added
- Scientific Data (Nature) integration
- Zenodo repository support
- GEO (Gene Expression Omnibus) integration
- ChEMBL pharmacology database support
- ClinicalTrials.gov integration
- Domain-based module organization (neuroscience, biology, general, medical, pharmacology)
- Database build and search across all sources

### Changed
- Reorganized package structure into domain submodules
- Improved unified search across all repositories
- Updated CLI with domain-specific commands

## [0.2.0] - 2026-02-15

### Added
- Local database indexing for fast searching
- Batch fetching improvements
- Enhanced metadata parsing

### Fixed
- Rate limiting for API calls
- Pagination handling for large result sets

## [0.1.0] - 2026-01-29

### Added
- Initial release
- OpenNeuro dataset fetching and search
- DANDI Archive integration
- PhysioNet database support
- Unified search across repositories
- CLI with `openneuro`, `dandi`, `physionet`, `search` commands
- MCP server with 8 tools for AI agent integration
- Database module for local caching
- SciTeX session integration support

### MCP Tools
- `dataset_openneuro_fetch` - Fetch OpenNeuro datasets
- `dataset_openneuro_search` - Search OpenNeuro
- `dataset_dandi_fetch` - Fetch DANDI datasets
- `dataset_dandi_search` - Search DANDI
- `dataset_physionet_fetch` - Fetch PhysioNet datasets
- `dataset_physionet_search` - Search PhysioNet
- `dataset_search` - Unified cross-repository search
- `dataset_stats` - Repository statistics
