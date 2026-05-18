---
description: |
  [TOPIC] Installation
  [DETAILS] pip install scitex-dataset. Pulls httpx + click + scitex-dev + scitex-config. Optional [mcp] adds MCP server.
tags: [scitex-dataset-installation]
---

# Installation

## Standard

```bash
pip install scitex-dataset
```

Pulls `httpx>=0.24`, `click>=8.0`, `scitex-dev`, and `scitex-config`.

## Optional extras

| Extra | Purpose |
|---|---|
| `mcp` | MCP server (`scitex-dataset mcp serve`) |
| `scitex` | Pull in the `scitex` umbrella for full ecosystem integration |
| `dev` | Test + lint tooling |
| `docs` | Sphinx + RTD theme |
| `all` | Everything above |

```bash
pip install 'scitex-dataset[mcp]'
```

## Verify

```bash
python -c "import scitex_dataset; print(scitex_dataset.__version__)"
scitex-dataset --version
scitex-dataset --help
```

## Editable install (development)

```bash
git clone https://github.com/ywatanabe1989/scitex-dataset
cd scitex-dataset
pip install -e '.[dev]'
```

## Umbrella alternative

`pip install scitex` exposes the same module as `scitex.dataset`.
