#!/bin/bash
# Run LaMP-QA (3 categories × 800-900 items, rubric-based 0-2 scoring) for a
# single served model, using the native multibench `lampqa` runner.
#
# Required env:
#   MODEL        - friendly output name
#   PORT         - local vLLM port of the simulator
# Optional env:
#   MODEL_ID     - actual served id (default: $MODEL)
#   JUDGE_MODEL  - judge model id (default: $MODEL_ID)
#   JUDGE_PORT   - judge port (default: $PORT)
#   WORKERS      - concurrency (default 32)
#   TOPK         - BM25 top-k profile snippets (default 5)
#   LIMIT        - items cap per category (default: unlimited)
#   CATEGORIES   - space-separated categories subset (default: all 3)
#   SPLIT        - train / validation / test (default: test)
#   OUT_ROOT     - results root
set -uo pipefail

: "${MODEL:?set MODEL}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-32}"
TOPK="${TOPK:-5}"
SPLIT="${SPLIT:-test}"
JUDGE_PORT="${JUDGE_PORT:-$PORT}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_ID}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"
JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1"

extra=()
[ -n "${LIMIT:-}" ] && extra+=(--max-items "$LIMIT")
if [ -n "${CATEGORIES:-}" ]; then
    read -r -a CATS <<< "$CATEGORIES"
    extra+=(--categories "${CATS[@]}")
fi

cd "$REPO"
echo "[lampqa] sim=$MODEL_ID @:$PORT  judge=$JUDGE_MODEL @:$JUDGE_PORT  workers=$WORKERS topk=$TOPK split=$SPLIT"
"$PY" -m multibench.cli run lampqa -- \
    --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
    --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
    --topk "$TOPK" --split "$SPLIT" \
    --output-dir "$OUT/LaMP-QA" \
    "${extra[@]}" \
    2>&1 | tee "$LOGDIR/lampqa.log"
echo "[lampqa] done $(date)"
