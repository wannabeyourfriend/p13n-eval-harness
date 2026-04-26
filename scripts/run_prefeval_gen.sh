#!/bin/bash
# Run PrefEval --stage gen (generation only, no judge yet) for a single model
# across all 20 topics. Judge step runs separately on Mac with OpenAI access.
set -uo pipefail

: "${MODEL:?set MODEL}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-16}"
INTER_TURNS="${INTER_TURNS:-2}"
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
echo "[prefeval-gen] model=$MODEL port=$PORT workers=$WORKERS topics=${#TOPICS_ARR[@]}"
for topic in "${TOPICS_ARR[@]}"; do
    "$PY" -m multibench.cli run prefeval -- \
        --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
        --topic "$topic" --inter-turns "$INTER_TURNS" \
        --task zero-shot --stage gen --pref-form explicit \
        --output-dir "$OUT/PrefEval" \
        2>&1 | tee -a "$LOGDIR/prefeval_gen.log"
done
echo "[prefeval-gen] done $(date)"
