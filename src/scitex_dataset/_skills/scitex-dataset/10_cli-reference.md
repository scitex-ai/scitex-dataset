---
description: |
  [TOPIC] CLI examples
  [DETAILS] Entry points into the canonical domain/source/action reference.
tags: [scitex-dataset-cli-reference, scitex-dataset]
---


# CLI examples

Use [04_cli-reference.md](04_cli-reference.md) as the canonical command and
option reference. Catalog requests use `<domain> <source> fetch`; benchmark
commands use the `ai-for-science` domain.

```bash
scitex-dataset neuroscience openneuro fetch --help
scitex-dataset general zenodo fetch --help
scitex-dataset ai-for-science corebench --help
scitex-dataset db show-stats --help
scitex-dataset completion install --help
```

There is no top-level `search`, `fetch`, `list-sources` or
`fetch_all_datasets` command. For index searches, use `db search` with a
configured store. The old `db stats` form reports its replacement and exits
with status 2; use `db show-stats`.
