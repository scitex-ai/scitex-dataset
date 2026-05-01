#!/bin/bash
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-dataset/examples/00_run_all.sh
# Run all scitex-dataset examples sequentially with a tee'd log.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$SCRIPT_DIR/$(basename "$0").log"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "$1" | tee -a "$LOG_PATH"; }

usage() {
    echo "Usage: $(basename "$0") [-h]"
    echo ""
    echo "Run all scitex-dataset examples sequentially. Tees output to"
    echo "$(basename "$0").log next to this script."
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

cd "$SCRIPT_DIR"
echo >"$LOG_PATH"

log "========================================"
log "scitex-dataset Examples Runner"
log "========================================"
log ""

mapfile -t SCRIPTS < <(printf '%s\n' 01_*.py 02_*.py 03_*.sh 04_*.py | sort)
TOTAL=${#SCRIPTS[@]}
COUNT=0
PASSED=0
FAILED=0

for script in "${SCRIPTS[@]}"; do
    [[ -f "$script" ]] || continue
    COUNT=$((COUNT + 1))
    log ""
    log "[$COUNT/$TOTAL] Running $script..."
    if [[ "$script" == *.py ]]; then
        if python "$script" >>"$LOG_PATH" 2>&1; then
            log "${GREEN}[PASS]${NC} $script"
            PASSED=$((PASSED + 1))
        else
            log "${RED}[FAIL]${NC} $script (see $LOG_PATH)"
            FAILED=$((FAILED + 1))
        fi
    else
        if bash "$script" >>"$LOG_PATH" 2>&1; then
            log "${GREEN}[PASS]${NC} $script"
            PASSED=$((PASSED + 1))
        else
            log "${RED}[FAIL]${NC} $script (see $LOG_PATH)"
            FAILED=$((FAILED + 1))
        fi
    fi
done

log ""
log "========================================"
log "Results: $PASSED passed, $FAILED failed"
log "========================================"
log ""
log "Log: $LOG_PATH"

# EOF
