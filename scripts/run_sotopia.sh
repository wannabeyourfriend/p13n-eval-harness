#!/bin/bash
# Run Sotopia (90 scenarios, 7-dimension judge) for a single served model.
#
# Required env:
#   MODEL         - model name acting as the simulator
#   PORT          - local vLLM port of the simulator
# Optional env:
#   JUDGE_MODEL   - judge model name (default: $MODEL, self-judge)
#   JUDGE_PORT    - judge port (default: $PORT)
#   WORKERS       - concurrency (default 32)
#   MAX_SCENARIOS - cap (default: all 90)
#   OUT_ROOT      - results root
#   PY            - python interpreter
#
# Sotopia needs a judge. By default we use the same endpoint (self-judge).
# For better quality, point --judge-* at a stronger model.
set -uo pipefail

: "${MODEL:?set MODEL (friendly output name)}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_ID}"
JUDGE_PORT="${JUDGE_PORT:-$PORT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"
JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1"

extra_args=()
[ -n "${MAX_SCENARIOS:-}" ] && extra_args+=(--max-scenarios "$MAX_SCENARIOS")

cd "$REPO"
echo "[sotopia] sim=$MODEL@:$PORT  judge=$JUDGE_MODEL@:$JUDGE_PORT  workers=$WORKERS"
"$PY" -m multibench.cli run sotopia -- \
    --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
    --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
    --output-dir "$OUT/Sotopia" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOGDIR/sotopia.log"
echo "[sotopia] done $(date)"
