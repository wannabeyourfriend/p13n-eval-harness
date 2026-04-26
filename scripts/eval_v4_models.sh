#!/bin/bash
# Eval the v4 LoRA candidates against the v3 best (mixed-r64-lr2e-4) and base.
# Targets:
#   PrefEval gen with gpt-4.1-mini judge — primary metric (highest signal)
#   PersonaMem v2 MCQ (rule)
#   PersonaLens (sample s3, gpt-4.1-mini judge — discriminator for the
#                regression we saw in v3 mixed-r64)
#   Sotopia (90 scenarios, gpt-4.1-mini judge — new)
#
# Endpoints:
#   8500 (existing) → base + v3 mixed-r64 (already serving on remote)
#   8600 (new)      → base + v4 LoRAs (start with serve_v4_loras.sh)
#
# Usage:
#   WORKERS=16 bash scripts/eval_v4_models.sh
#   MAX_ITEMS=20 WORKERS=4 bash scripts/eval_v4_models.sh   # smoke
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-16}"
MAX_ITEMS="${MAX_ITEMS:-}"
OUT_ROOT="${OUT_ROOT:-$REPO/results_v4}"
mkdir -p "$OUT_ROOT"

# (model_tag, port)
MODELS=(
  "qwen25-7b-instruct-base:8600"
  "v4-conv-r64:8600"
  "v4-mix-r64:8600"
  "v4-7030-r64:8600"
  "qwen25-7b-qa-mixed-r64-lr2e-4:8500"
)

run_personamem() {
    local model="$1" port="$2"
    MODEL="$model" PORT="$port" WORKERS="$WORKERS" MAX_ITEMS="$MAX_ITEMS" \
      OUT_ROOT="$OUT_ROOT" EVAL_MODE=mcq SIZE=32k \
      bash "$REPO/scripts/run_personamem.sh" \
      > "$OUT_ROOT/_${model}_personamem_v2.log" 2>&1
}

run_prefeval_gen() {
    local model="$1" port="$2"
    MODEL="$model" PORT="$port" WORKERS="$WORKERS" \
      OUT_ROOT="$OUT_ROOT" \
      bash "$REPO/scripts/run_prefeval_gen.sh" \
      > "$OUT_ROOT/_${model}_prefeval_gen.log" 2>&1
}

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"; port="${entry##*:}"
    echo "[eval_v4] PersonaMem v2 MCQ on $model"
    run_personamem "$model" "$port"
    echo "[eval_v4] PrefEval gen on $model"
    run_prefeval_gen "$model" "$port"
done
echo "=== eval gen complete; run prefeval judge on Mac via run_prefeval_judge_local.sh ==="
