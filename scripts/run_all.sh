#!/bin/bash
# Run all 6 benchmarks for a single model against a single vLLM endpoint.
#
# Usage:
#   MODEL=us-profile-mar31 API_BASE=http://localhost:8002/v1 \
#   JUDGE_API_BASE=https://api.openai.com/v1 JUDGE_MODEL=gpt-4.1-mini \
#   ./scripts/run_all.sh
set -uo pipefail

MODEL="${MODEL:?set MODEL env var (e.g. us-profile-mar31)}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
JUDGE_API_BASE="${JUDGE_API_BASE:-$API_BASE}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
WORKERS="${WORKERS:-64}"
OUT="${OUT:-results/${MODEL}}"
MAX_ITEMS="${MAX_ITEMS:-}"

common=(--api-base "$API_BASE" --model "$MODEL" --workers "$WORKERS")
[ -n "$MAX_ITEMS" ] && common+=(--max-items "$MAX_ITEMS")

echo "== BigTom =="
multibench run bigtom -- "${common[@]}" --condition both \
    --output-dir "$OUT/BigTom"

echo "== LaMP (LaMP-1..7) =="
for t in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7; do
    multibench run lamp -- "${common[@]}" --task "$t" --use-profile \
        --num-retrieved 3 --retriever bm25 \
        --output-dir "$OUT/LaMP/${t}"
done

echo "== PrefEval (all 20 topics) =="
TOPICS=(travel_transportation shop_motors lifestyle_beauty travel_restaurant
        shop_fashion entertain_shows pet_ownership lifestyle_fit entertain_games
        shop_home lifestyle_health travel_activities education_learning_styles
        entertain_music_book professional_work_location_style education_resources
        lifestyle_dietary shop_technology travel_hotel entertain_sports)
for topic in "${TOPICS[@]}"; do
    multibench run prefeval -- "${common[@]}" --topic "$topic" \
        --inter-turns 2 --task zero-shot --stage all \
        --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
        --output-dir "$OUT/PrefEval"
done

echo "== PersonaLens =="
multibench run personalens -- --stage gen "${common[@]}" --sample s5 \
    --output-dir "$OUT/PersonaLens"
for dim in personalization task_completion; do
    multibench run personalens -- --stage eval \
        --api-base "$JUDGE_API_BASE" --model "$JUDGE_MODEL" --workers "$WORKERS" \
        --model-tag "${MODEL}_d_p_s" --eval-dim "$dim" --sample s5 \
        --output-dir "$OUT/PersonaLens"
done

echo "== PersonaMem-v2 =="
multibench run personamem -- "${common[@]}" --eval-mode mcq --size 32k \
    --output-dir "$OUT/PersonaMem"

echo "== Sotopia =="
multibench run sotopia -- "${common[@]}" \
    --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
    --output-dir "$OUT/Sotopia"

echo "=== DONE $(date) ==="
