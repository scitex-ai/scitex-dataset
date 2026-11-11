---
description: |
  [TOPIC] CLI reference
  [DETAILS] scitex-dataset uses the SciTeX 3-level noun-verb grammar:
  domain → source → action.
tags: [scitex-dataset-cli-reference]
---

# CLI Reference

```text
scitex-dataset <domain> <source> <action> [OPTIONS]
```

Source groups sit under their scientific domain, with the action last.
Global utility groups are listed below.

## Domains and datasets

```text
neuroscience:
  - openneuro       BIDS neuroimaging
  - dandi           NWB neurophysiology
  - physionet       EEG / ECG / waveforms
  - gin             G-Node dataset catalog
general:
  - zenodo          general scientific repository
  - figshare        research data sharing
  - openml          ML datasets
  - huggingface     ML datasets/models (on-demand)
biology:
  - geo             Gene Expression Omnibus
pharmacology:
  - moleculenet     molecular ML benchmarks
  - chembl          bioactivity database
medical:
  - clinicaltrials  clinical study registry
ai-for-science:
  - corebench
  - bixbench
  - biomysterybench
```

The 11 catalog sources expose `fetch` with `-n / --max-datasets`,
`-o / --output` and `-v / --verbose`. Zenodo and Figshare also accept
`-q / --query`; OpenNeuro accepts `-b / --batch-size`.

HuggingFace has a separate `fetch REPO_ID` command with `--local-dir`,
`--repo-type` and `--max-workers`, plus `search`, `info` and `download-file`.
The three `ai-for-science` benchmarks expose `download`, `prepare`,
`standardize`, `validate` and `score`, rather than catalog `fetch`.

## Global flags

| Flag | Purpose |
|---|---|
| `-V`, `--version` | Show version and exit |
| `--help-recursive` | Show help for all commands |
| `--json` | Accepted by the root parser; check the selected command's output options |
| `-h`, `--help` | Show this message and exit |

## Configuration precedence

The `_config.find_config_path(cli_path=...)` Python helper checks:

1. An explicit `cli_path` argument
2. `$SCITEX_DATASET_CONFIG` env var
3. `<project>/.scitex/dataset/config.yaml`  (project scope wins)
4. `$SCITEX_DIR/dataset/config.yaml` (default `~/.scitex/dataset/config.yaml`)

Runtime files (snapshots, caches, logs) live under
`<scope-root>/runtime/`. The dataset index does not: it lives in the shared
SciTeX store, which `SCITEX_STORE_DSN` repoints and `db show-stats` names.

The root CLI currently does **not** register `--config`, despite the config
chain printed in its help. Catalog commands do not call `load_config`.
Use the source-specific CLI flags; do not assume the resolver configures
those requests. This is an unresolved CLI integration gap, not an available
global option. Likewise, pass `--json` to a command that explicitly supports
it, such as `db show-stats --json`; the root flag is not a guarantee of
structured output from every command.

## Examples

```bash
scitex-dataset neuroscience openneuro fetch -n 50
scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
scitex-dataset general huggingface search "biology" -n 20
scitex-dataset pharmacology chembl fetch
scitex-dataset db build
scitex-dataset db search "Alzheimer"
scitex-dataset db show-stats
scitex-dataset db show-stats --json
scitex-dataset completion install --shell bash
scitex-dataset completion status --json
```

## Top-level groups (alphabetical)

| Command | Purpose |
|---|---|
| `<domain>` (6 groups) | Domain → source → action chain |
| `completion` | Install or inspect a completion drop-in (`install`, `status`) |
| `db` | Full-text dataset index (`build`, `search`, `show-stats`, `clear`) |
| `docs` | Browse package documentation |
| `list-python-apis` | Enumerate the public Python API tree |
| `mcp` | MCP server commands |
| `print-shell-completion` | Emit the completion script for a selected shell |
| `install-shell-completion` | Install the completion drop-in |
| `skills` | Browse and install the package's bundled skills |

## Legacy aliases (hidden, redirect with status 2)

| Old form | New form |
|---|---|
| `fetch-openneuro [...]` | `neuroscience openneuro fetch [...]` |
| `fetch-<source>` | `<domain> <source> fetch` |
| `<source>` (bare) | `<domain> <source> fetch` |
| `hf <verb>` | `general huggingface <verb>` |
| `hf-<verb>` | `general huggingface <verb>` |
| `db stats` | `db show-stats` |

For per-command flags, run ``scitex-dataset <command> --help`` or
``scitex-dataset --help-recursive``.

## See also

- [10_cli-reference.md](10_cli-reference.md) — historical CLI notes
- [11_mcp-tools.md](11_mcp-tools.md) — MCP tool equivalents
- [12_env-vars.md](12_env-vars.md) — env vars read at runtime
- [14_data-sources.md](14_data-sources.md) — per-source quick-start

## Source and checks

The registry is owned by [`_sources.py`](../../_sources.py). Command
registration and options live in [`_cli/__init__.py`](../../_cli/__init__.py),
[`_groups.py`](../../_cli/_groups.py), [`_factory.py`](../../_cli/_factory.py)
and [`_completion.py`](../../_cli/_completion.py). Configuration resolution
is separate in [`_config.py`](../../_config.py).

Existing controls cover [domain and source groups](../../../../tests/scitex_dataset/_cli/test__groups.py)
and [completion behavior](../../../../tests/scitex_dataset/_cli/test__completion_runtime.py).
Command parsing and help checks do not establish remote fetch availability,
store access or benchmark correctness.
