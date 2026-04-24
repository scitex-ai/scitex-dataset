---
name: scitex-dataset-env-vars
description: Environment variables read by scitex-dataset at import / runtime. Follow SCITEX_<MODULE>_* convention — see general/10_arch-environment-variables.md.
---

# scitex-dataset — Environment Variables

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `SCITEX_DATASET_CACHE_DIR` | Override for the local dataset cache (OpenNeuro / DANDI / PhysioNet / GEO / ChEMBL / ClinicalTrials downloads). | `~/.scitex/dataset` | path |
| `SCITEX_DATASET_COMPLETE` | Internal sentinel: set when the standalone is importable (used by the umbrella shim). | unset | bool (presence) |

## Notes

- `SCITEX_DATASET_CACHE_DIR` is user-facing; point it at a fast SSD or a scratch dir on HPC.
- `SCITEX_DATASET` (bare) appears only as a substring match inside longer var names — not a real env var.

## Audit

```bash
grep -rhoE 'SCITEX_[A-Z0-9_]+' $HOME/proj/scitex-dataset/src/ | sort -u
```
