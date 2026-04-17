#!/bin/bash
# Run PrefEval (20 topics), BigTom, and Sotopia for a single served model.
#
# Required env:
#   MODEL          - model name as served by vLLM (e.g. no-state-us)
#   PORT           - local vLLM port (e.g. 8001)
# Optional env:
#   WORKERS        - concurrency (default 32, recommended for 1 A100 + 7B vLLM)
#   OUT_ROOT       - results root (default: $REPO/results)
#   INTER_TURNS    - prefeval filler-turn count (default 5)
#   JUDGE_PORT     - port of judge model (default: $PORT, self-judge)
#   JUDGE_MODEL    - judge model name (default: $MODEL)
#   SKIP           - space-separated benchmark names to skip (e.g. "sotopia")
set -uo pipefail

: "${MODEL:?set MODEL (e.g. no-state-us)}"
: "${PORT:?set PORT (e.g. 8001)}"
WORKERS="${WORKERS:-32}"
INTER_TURNS="${INTER_TURNS:-5}"
JUDGE_PORT="${JUDGE_PORT:-$PORT}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SKIP="${SKIP:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

API_BASE="http://localhost:${PORT}/v1"
JUDGE_API_BASE="http://localhost:${JUDGE_PORT}/v1"
PY="${PYTHON:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"

common=(--api-base "$API_BASE" --model "$MODEL" --workers "$WORKERS")

banner() { echo "========== [$MODEL @ :$PORT] $1 =========="; }
skipped() { [[ " $SKIP " == *" $1 "* ]]; }

cd "$REPO"

# ---------------- BigTom ----------------
if ! skipped bigtom; then
    banner "BigTom"
    "$PY" -m multibench.cli run bigtom -- "${common[@]}" --condition both \
        --output-dir "$OUT/BigTom" \
        2>&1 | tee "$LOGDIR/bigtom.log"
fi

# ---------------- PrefEval (20 topics) ----------------
if ! skipped prefeval; then
    banner "PrefEval"
    TOPICS=(travel_transportation shop_motors lifestyle_beauty travel_restaurant
            shop_fashion entertain_shows pet_ownership lifestyle_fit entertain_games
            shop_home lifestyle_health travel_activities education_learning_styles
            entertain_music_book professional_work_location_style education_resources
            lifestyle_dietary shop_technology travel_hotel entertain_sports)
    for topic in "${TOPICS[@]}"; do
        "$PY" -m multibench.cli run prefeval -- "${common[@]}" \
            --topic "$topic" --inter-turns "$INTER_TURNS" \
            --task zero-shot --stage cls --pref-form explicit \
            --output-dir "$OUT/PrefEval" \
            2>&1 | tee -a "$LOGDIR/prefeval.log"
    done
fi

# ---------------- Sotopia ----------------
if ! skipped sotopia; then
    banner "Sotopia"
    "$PY" -m multibench.cli run sotopia -- "${common[@]}" \
        --judge-api-base "$JUDGE_API_BASE" --judge-model "$JUDGE_MODEL" \
        --output-dir "$OUT/Sotopia" \
        2>&1 | tee "$LOGDIR/sotopia.log"
fi

echo "[$MODEL] DONE $(date)"
