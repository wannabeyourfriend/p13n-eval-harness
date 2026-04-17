#!/bin/bash
# Run PrefEval + BigTom + Sotopia on 3 models in parallel.
#
# Endpoints (as configured):
#   8001  no-state-us        (GPU 1)
#   8002  us-profile-mar31   (GPU 2)
#   8003  oracle-profile-only (GPU 3)
#
# Each model runs its three benchmarks sequentially on its own GPU. The three
# model streams run in parallel since they hit independent ports/GPUs.
#
# Usage:
#   WORKERS=32 bash scripts/eval_3models.sh
#   WORKERS=32 SKIP="sotopia" bash scripts/eval_3models.sh          # skip sotopia
#   WORKERS=32 INTER_TURNS=2 bash scripts/eval_3models.sh           # match old runs
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-32}"
INTER_TURNS="${INTER_TURNS:-5}"
SKIP="${SKIP:-}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"

mkdir -p "$OUT_ROOT"

run_one() {
    local model="$1" port="$2"
    MODEL="$model" PORT="$port" WORKERS="$WORKERS" INTER_TURNS="$INTER_TURNS" \
        SKIP="$SKIP" OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/eval_one_model.sh" \
        > "$OUT_ROOT/_${model}.log" 2>&1
    echo "[done] $model"
}

echo "Launching 3 models in parallel. Tail individual logs at $OUT_ROOT/_<model>.log"

run_one no-state-us          8001 &
pid1=$!
run_one us-profile-mar31     8002 &
pid2=$!
run_one oracle-profile-only  8003 &
pid3=$!

wait "$pid1" "$pid2" "$pid3"
echo "=== all 3 models done $(date) ==="

echo; echo "=== SUMMARY ==="
/home/2025user/zhou/anaconda3/envs/persona/bin/python "$REPO/scripts/summarize.py" \
    --results-root "$OUT_ROOT" \
    --models no-state-us us-profile-mar31 oracle-profile-only
