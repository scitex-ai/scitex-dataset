---
description: |
  [TOPIC] CLI reference
  [DETAILS] scitex-dataset uses the SciTeX 3-level noun-verb grammar:
  domain → dataset → action.
tags: [scitex-dataset-cli-reference]
---

# CLI Reference

```
scitex-dataset <domain> <dataset> <action> [OPTIONS]
```

The CLI follows the SciTeX noun-verb grammar
(`general/03_interface_02_cli/02_subcommand-structure-noun-verb.md`).
Every leaf is a verb; every interior token is a noun.

## Domains and datasets

```
neuroscience:
  - openneuro       BIDS neuroimaging
  - dandi           NWB neurophysiology
  - physionet       EEG / ECG / waveforms
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
```

Every dataset exposes a ``fetch`` action with the standard flags
``-n / -o / -v`` (and ``-q`` where the upstream API supports a query).
HuggingFace adds ``search`` / ``info`` / ``download-file``.

## Global flags

| Flag | Purpose |
|---|---|
| `-V`, `--version` | Show version and exit |
| `--help-recursive` | Show help for all commands |
| `--json` | Emit machine-readable JSON (subcommands honour where applicable) |
| `-h`, `--help` | Show this message and exit |

## Configuration precedence

The same chain as every other SciTeX package (see
`general/01_ecosystem_06_local-state-directories`):

1. `--config <path>` CLI flag
2. `$SCITEX_DATASET_CONFIG` env var
3. `<project>/.scitex/dataset/config.yaml`  (project scope wins)
4. `$SCITEX_DIR/dataset/config.yaml` (default `~/.scitex/dataset/config.yaml`)

Runtime files (the local SQLite index, snapshots, logs) live under
`<scope-root>/runtime/`.

## Examples

```bash
scitex-dataset neuroscience openneuro fetch -n 50
scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
scitex-dataset general huggingface search "biology" -n 20
scitex-dataset pharmacology chembl fetch
scitex-dataset db build
scitex-dataset db search "Alzheimer"
scitex-dataset db show-stats
```

## Top-level groups (alphabetical)

| Command | Purpose |
|---|---|
| `<domain>` (5 groups) | Domain → dataset → action chain |
| `db` | Local SQLite + FTS5 index (`build`, `search`, `show-stats`, `clear`) |
| `docs` | Browse package documentation |
| `list-python-apis` | Enumerate the public Python API tree |
| `mcp` | MCP server commands |
| `print-tab-completion` | Emit shell completion script |

## Legacy aliases (hidden, redirect with status 2)

| Old form | New form |
|---|---|
| `fetch-openneuro [...]` | `neuroscience openneuro fetch [...]` |
| `fetch-<source>` | `<domain> <source> fetch` |
| `<source>` (bare) | `<domain> <source> fetch` |
| `hf <verb>` | `general huggingface <verb>` |
| `hf-<verb>` | `general huggingface <verb>` |
| `db stats` | `db show-stats` |
| `completion` | `print-tab-completion` |

For per-command flags, run ``scitex-dataset <command> --help`` or
``scitex-dataset --help-recursive``.

## See also

- [10_cli-reference.md](10_cli-reference.md) — historical CLI notes
- [11_mcp-tools.md](11_mcp-tools.md) — MCP tool equivalents
- [12_env-vars.md](12_env-vars.md) — env vars read at runtime
- [14_data-sources.md](14_data-sources.md) — per-source quick-start
