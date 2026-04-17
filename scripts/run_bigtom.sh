#!/bin/bash
# Run BigTom (both true_belief + false_belief) for a single served model.
#
# Required env:
#   MODEL    - model name
#   PORT     - local vLLM port
# Optional env:
#   WORKERS  - concurrency (default 32)
#   OUT_ROOT - results root (default: $REPO/results)
#   PY       - python interpreter
set -uo pipefail

: "${MODEL:?set MODEL (friendly output name)}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-32}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"

cd "$REPO"
echo "[bigtom] model=$MODEL port=$PORT workers=$WORKERS"
"$PY" -m multibench.cli run bigtom -- \
    --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
    --condition both --output-dir "$OUT/BigTom" \
    2>&1 | tee "$LOGDIR/bigtom.log"
echo "[bigtom] done $(date)"
