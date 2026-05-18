---
description: |
  [TOPIC] Env Vars
  [DETAILS] Every SCITEX_* / HF_* environment variable scitex-dataset
  reads at runtime, with the file:line site of the read.
tags: [scitex-dataset-env-vars, scitex-dataset]
---

# scitex-dataset — Environment Variables

| Variable | Purpose | Default | Type | Read at |
|---|---|---|---|---|
| `SCITEX_DIR` | Relocates the user-scope state root for every scitex-* package. ``$SCITEX_DIR/dataset/config.yaml`` is the user-scope config, and ``$SCITEX_DIR/dataset/runtime/datasets.db`` is the local SQLite + FTS5 index when no project-scope override is in scope. | `~/.scitex` | path | ``_config.py``, ``database.py`` |
| `SCITEX_DATASET_CONFIG` | Explicit override for the config file path (CLI flag still wins). | unset | path | ``_config.py:find_config_path`` |
| `_SCITEX_DATASET_COMPLETE` | Click's tab-completion sentinel; set transiently by ``scitex-dataset print-tab-completion`` to render the bash/zsh/fish hook script. Not user-facing. | unset | bool (presence) | ``_cli/__init__.py`` |
| `HF_TOKEN` | HuggingFace Hub token (direct value). Highest priority for gated repos. | unset | string | ``general/huggingface.py:_resolve_token`` |
| `HF_TOKEN_PATH` | Path to a file containing the HuggingFace token. Read when ``HF_TOKEN`` is unset. Falls back to ``~/.bash.d/secrets/access_tokens/huggingface.txt`` if neither is set. | unset | path | ``general/huggingface.py:_resolve_token`` |
| `HF_HOME` | HuggingFace's own cache directory. ``hf fetch --hf-home <path>`` overrides it transiently for one snapshot download. | unset (HF default) | path | ``general/huggingface.py:fetch_dataset`` |

## Configuration precedence

The CLI honors the SciTeX local-state-directories chain (highest first):

1. ``--config <path>`` CLI flag
2. ``$SCITEX_DATASET_CONFIG`` env var (this file)
3. ``<project>/.scitex/dataset/config.yaml`` (project scope)
4. ``$SCITEX_DIR/dataset/config.yaml`` — default ``~/.scitex/dataset/config.yaml`` (user scope)

Project scope wins over user scope. ``SCITEX_DIR`` relocates the user
scope atomically without touching the project scope. See
``general/01_ecosystem_06_local-state-directories``.

## Spartan / HPC tip

For large HuggingFace fetches on Spartan, point ``--hf-home`` (or
``HF_HOME``) at the project filesystem so the content-addressed cache
doesn't grow your home quota:

```bash
scitex-dataset general huggingface fetch <repo_id> \
    --hf-home /data/gpfs/projects/punim2354/.hf-cache \
    -d /data/gpfs/projects/punim2354/<repo_dir>
```

## Audit

```bash
rg -nhoE 'SCITEX_[A-Z0-9_]+|HF_TOKEN[A-Z_]*|HF_HOME|SCITEX_DIR' \
    ~/proj/scitex-dataset/src/ | sort -u
```

If the rg output lists an env var not present in the table above, the
table is stale.
