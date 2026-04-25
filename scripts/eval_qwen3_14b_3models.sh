#!/bin/bash
# Eval the 3 Qwen3-14B ablations across all 4 target benchmarks
# (bigtom, prefeval, lamp, personamem) in parallel — one model per GPU.
#
# Endpoints (must be running, see training/scripts/serve_qwen3_14b_*.sh):
#   8001  qwen3-14b-no-state-us            (GPU 1)
#   8002  qwen3-14b-oracle-profile-only    (GPU 2)
#   8003  qwen3-14b-us-profile-mar31       (GPU 3)
#
# Usage:
#   WORKERS=32 bash scripts/eval_qwen3_14b_3models.sh
#   WORKERS=32 MAX_ITEMS=20 bash scripts/eval_qwen3_14b_3models.sh   # smoke
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-32}"
INTER_TURNS="${INTER_TURNS:-2}"
MAX_ITEMS="${MAX_ITEMS:-}"
SKIP="${SKIP:-sotopia}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
mkdir -p "$OUT_ROOT"

run_one() {
    local model="$1" port="$2"
    MODEL="$model" PORT="$port" WORKERS="$WORKERS" INTER_TURNS="$INTER_TURNS" \
        MAX_ITEMS="$MAX_ITEMS" SKIP="$SKIP" OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/eval_one_model.sh" \
        > "$OUT_ROOT/_${model}_core.log" 2>&1 &
    local p1=$!

    MODEL="$model" PORT="$port" WORKERS="$WORKERS" MAX_ITEMS="$MAX_ITEMS" \
        OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/run_lamp.sh" \
        > "$OUT_ROOT/_${model}_lamp.log" 2>&1 &
    local p2=$!

    MODEL="$model" PORT="$port" WORKERS="$WORKERS" MAX_ITEMS="$MAX_ITEMS" \
        OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/run_personamem.sh" \
        > "$OUT_ROOT/_${model}_personamem.log" 2>&1 &
    local p3=$!

    wait "$p1" "$p2" "$p3"
    echo "[done] $model"
}

echo "Launching 3 Qwen3-14B models in parallel."
run_one qwen3-14b-no-state-us         8001 &
PID1=$!
run_one qwen3-14b-oracle-profile-only 8002 &
PID2=$!
run_one qwen3-14b-us-profile-mar31    8003 &
PID3=$!
wait "$PID1" "$PID2" "$PID3"
echo "=== all 3 Qwen3-14B models done $(date) ==="

echo; echo "=== SUMMARY ==="
/home/2025user/zhou/anaconda3/envs/persona/bin/python "$REPO/scripts/summarize.py" \
    --results-root "$OUT_ROOT" \
    --models qwen3-14b-no-state-us qwen3-14b-oracle-profile-only qwen3-14b-us-profile-mar31
