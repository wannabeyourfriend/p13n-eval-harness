#!/bin/bash
# Orchestrator: run BigTom → PrefEval → Sotopia for a single (MODEL, PORT).
# Delegates each benchmark to its own dedicated script so they can also be
# invoked individually.
#
# Required env:
#   MODEL
#   PORT
# Optional env (passed through):
#   WORKERS INTER_TURNS TOPICS MAX_SCENARIOS JUDGE_MODEL JUDGE_PORT OUT_ROOT PY
#   SKIP   - space-separated benchmark names to skip: "bigtom" / "prefeval" /
#            "sotopia" (e.g. SKIP="bigtom sotopia")
set -uo pipefail

: "${MODEL:?set MODEL}"
: "${PORT:?set PORT}"
SKIP="${SKIP:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

skipped() { [[ " $SKIP " == *" $1 "* ]]; }

run_bm() {
    local name="$1"; shift
    if skipped "$name"; then
        echo "[$MODEL @ :$PORT] SKIP $name"
        return 0
    fi
    echo "========== [$MODEL @ :$PORT] $name =========="
    "$@"
}

run_bm bigtom   bash "$REPO/scripts/run_bigtom.sh"
run_bm prefeval bash "$REPO/scripts/run_prefeval.sh"
run_bm sotopia  bash "$REPO/scripts/run_sotopia.sh"

echo "[$MODEL] ALL DONE $(date)"
