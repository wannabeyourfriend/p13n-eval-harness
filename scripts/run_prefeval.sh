#!/bin/bash
# Run PrefEval classification over all 20 topics for a single served model.
#
# Required env:
#   MODEL        - model name as served (e.g. no-state-us, qwen2.5-7b-instruct)
#   PORT         - local vLLM port
# Optional env:
#   WORKERS      - concurrency (default 32 for 1 A100 + 7B vLLM)
#   INTER_TURNS  - filler-turn count (default 5)
#   TOPICS       - space-separated subset of topics (default: all 20)
#   OUT_ROOT     - results root (default: $REPO/results)
#   PY           - python interpreter (default: persona env)
#
# Each topic writes results/$MODEL/PrefEval/mcq_results/explicit/zero-shot/<topic>/
set -uo pipefail

: "${MODEL:?set MODEL (friendly output name)}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"   # id actually served by vLLM
WORKERS="${WORKERS:-32}"
INTER_TURNS="${INTER_TURNS:-5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"

DEFAULT_TOPICS=(travel_transportation shop_motors lifestyle_beauty travel_restaurant
                shop_fashion entertain_shows pet_ownership lifestyle_fit entertain_games
                shop_home lifestyle_health travel_activities education_learning_styles
                entertain_music_book professional_work_location_style education_resources
                lifestyle_dietary shop_technology travel_hotel entertain_sports)
if [ -n "${TOPICS:-}" ]; then
    read -r -a TOPICS_ARR <<< "$TOPICS"
else
    TOPICS_ARR=("${DEFAULT_TOPICS[@]}")
fi

cd "$REPO"
echo "[prefeval] model=$MODEL port=$PORT workers=$WORKERS inter_turns=$INTER_TURNS topics=${#TOPICS_ARR[@]}"
for topic in "${TOPICS_ARR[@]}"; do
    "$PY" -m multibench.cli run prefeval -- \
        --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
        --topic "$topic" --inter-turns "$INTER_TURNS" \
        --task zero-shot --stage cls --pref-form explicit \
        --output-dir "$OUT/PrefEval" \
        2>&1 | tee -a "$LOGDIR/prefeval.log"
done
echo "[prefeval] done $(date)"
