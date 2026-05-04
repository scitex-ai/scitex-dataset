---
description: |
  [TOPIC] CLI reference
  [DETAILS] `scitex-dataset` console entry — search, fetch from per-source backends, list-sources, db build/search, mcp.
tags: [scitex-dataset-cli-reference]
---

# CLI Reference

```
scitex-dataset [OPTIONS] COMMAND [ARGS]...
```

Cross-repository dataset discovery — one CLI over OpenNeuro, DANDI,
PhysioNet, Zenodo, Scientific Data, GEO, ChEMBL, and ClinicalTrials.gov.

## Global options

| Flag | Purpose |
|---|---|
| `-V`, `--version` | Show version and exit |
| `--help-recursive` | Show help for all commands |
| `--json` | Emit machine-readable JSON |
| `-h`, `--help` | Show this message and exit |

## Configuration precedence

```
config.yaml -> $SCITEX_DATASET_CONFIG -> ~/.scitex/dataset/config.yaml -> defaults
```

## Commands (high level)

| Command | Purpose |
|---|---|
| `search` | Search across all configured sources |
| `fetch <source> <id>` | Fetch a dataset from a specific source |
| `list-sources` | Enumerate available sources |
| `db build` | Build the local dataset index |
| `db search` | Query the local index |
| `db stats` | Show local-DB statistics |
| `mcp serve` | Start the MCP server (requires `[mcp]`) |

## Examples

```bash
scitex-dataset list-sources
scitex-dataset search "EEG epilepsy"
scitex-dataset fetch openneuro ds003104
scitex-dataset db build
scitex-dataset db search "Alzheimer"
```

For per-command flags, run `scitex-dataset <command> --help` or
`scitex-dataset --help-recursive`.

## See also

- [10_cli-reference.md](10_cli-reference.md) — historical CLI notes
- [11_mcp-tools.md](11_mcp-tools.md) — MCP tool equivalents
- [12_env-vars.md](12_env-vars.md) — `SCITEX_*` env vars read at runtime
