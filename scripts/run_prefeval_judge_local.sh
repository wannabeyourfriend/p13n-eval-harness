#!/bin/bash
# Run PrefEval --stage judge locally on Mac with gpt-4.1-mini judge over OpenAI.
# Reads gen results from OUT_ROOT/<MODEL>/PrefEval/generation_results/...
set -uo pipefail

: "${MODEL:?set MODEL}"
WORKERS="${WORKERS:-8}"
INTER_TURNS="${INTER_TURNS:-2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results_prefeval_gen}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY not set" >&2
  exit 2
fi

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
echo "[prefeval-judge] model=$MODEL judge=$JUDGE_MODEL workers=$WORKERS topics=${#TOPICS_ARR[@]}"
for topic in "${TOPICS_ARR[@]}"; do
    uv run python -m multibench.cli run prefeval -- \
        --api-base http://placeholder/v1 --model "$MODEL" --workers "$WORKERS" \
        --topic "$topic" --inter-turns "$INTER_TURNS" \
        --task zero-shot --stage judge --pref-form explicit \
        --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
        --output-dir "$OUT/PrefEval" \
        2>&1 | tee -a "$LOGDIR/prefeval_judge.log"
done
echo "[prefeval-judge] done $(date)"
