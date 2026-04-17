#!/bin/bash
# Run LaMP-1..7 (excluding LaMP-6 — not in data) for a single served model.
#
# Required env:
#   MODEL        - friendly output name
#   PORT         - local vLLM port
# Optional env:
#   MODEL_ID     - actual served id (default: $MODEL)
#   WORKERS      - concurrency (default 32)
#   TASKS        - space-separated subset of LaMP tasks (default: 1 2 3 4 5 7)
#   RETRIEVER    - bm25 / contriever / recency / random (default: bm25)
#   NUM_RETRIEVED- top-k user history (default: 3)
#   OUT_ROOT     - results root
set -uo pipefail

: "${MODEL:?set MODEL}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-32}"
RETRIEVER="${RETRIEVER:-bm25}"
NUM_RETRIEVED="${NUM_RETRIEVED:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"

# transformers (pulled in by lamp runner for tokenizer trimming) pulls PIL which
# requires a newer libstdc++ than the host — preload the conda one.
export LD_PRELOAD="${LD_PRELOAD:-/home/2025user/zhou/anaconda3/envs/persona/lib/libstdc++.so.6}"

DEFAULT_TASKS=(LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7)
if [ -n "${TASKS:-}" ]; then
    read -r -a TASKS_ARR <<< "$TASKS"
else
    TASKS_ARR=("${DEFAULT_TASKS[@]}")
fi

cd "$REPO"
echo "[lamp] model=$MODEL port=$PORT tasks=${TASKS_ARR[*]} workers=$WORKERS"
for task in "${TASKS_ARR[@]}"; do
    "$PY" -m multibench.cli run lamp -- \
        --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
        --task "$task" --use-profile \
        --num-retrieved "$NUM_RETRIEVED" --retriever "$RETRIEVER" \
        --output-dir "$OUT/LaMP/$task" \
        2>&1 | tee -a "$LOGDIR/lamp.log"
done
echo "[lamp] done $(date)"
