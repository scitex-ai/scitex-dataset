#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-29 22:27:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-dataset/examples/03_cli_usage.sh

# Example: CLI usage for scitex-dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/03_cli_usage_out"

usage() {
    echo "Usage: $(basename "$0") [-h]"
    echo ""
    echo "Demonstrate scitex-dataset CLI commands."
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
}

case "${1:-}" in
-h | --help)
    usage
    exit 0
    ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "=== CLI Help ==="
scitex-dataset --help

echo
echo "=== OpenNeuro Help ==="
scitex-dataset neuroscience openneuro fetch --help

echo
echo "=== Fetching 10 datasets ==="
scitex-dataset neuroscience openneuro fetch -n 10 -o "$OUTPUT_DIR/datasets.json" -v

echo
echo "=== HuggingFace search (network optional) ==="
scitex-dataset general huggingface search "biology" -n 5 || true

echo
echo "Output saved to $OUTPUT_DIR/datasets.json"

# EOF
