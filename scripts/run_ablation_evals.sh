#!/usr/bin/env bash
# Evaluate base / privileged-student / blind-student on the paper's benchmarks.
#
# Serves ONE vLLM endpoint hosting the base model plus both LoRA adapters, so
# all three arms see an identical serving stack, then runs each benchmark per arm.
#
#   GPU=5 PORT=8011 bash scripts/run_ablation_evals.sh
#
# Env: BENCHES="bigtom personamem prefeval"  (default), GPU_UTIL=0.42
set -uo pipefail

GPU="${GPU:-5}"
PORT="${PORT:-8011}"
GPU_UTIL="${GPU_UTIL:-0.42}"
BENCHES="${BENCHES:-bigtom personamem prefeval}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_OUT="$HOME/mind2dialogue/training/outputs"
BASE="${BASE_MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
PY="${PY:-$REPO/.venv/bin/python}"
export LD_PRELOAD=""
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"

mkdir -p "$REPO/results" "$REPO/logs"

echo "[serve] starting vLLM on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES="$GPU" "$HOME/mind2dialogue/training/.venv/bin/vllm" serve "$BASE" \
  --served-model-name base \
  --enable-lora --max-lora-rank 64 --max-loras 2 \
  --lora-modules "privileged=$TRAIN_OUT/abl_privileged_r64_lr2e-4/final" \
                 "blind=$TRAIN_OUT/abl_blind_r64_lr2e-4/final" \
  --max-model-len 40000 --gpu-memory-utilization "$GPU_UTIL" \
  --chat-template-content-format string \
  --port "$PORT" > "$REPO/logs/vllm_ablation.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "[serve] waiting for readiness (pid $VLLM_PID)"
for _ in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "[serve] vLLM died, see logs/vllm_ablation.log"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null || { echo "[serve] timed out"; exit 1; }
echo "[serve] ready: $(curl -s http://localhost:${PORT}/v1/models | tr ',' '\n' | grep '"id"')"

for M in base privileged blind; do
  for B in $BENCHES; do
    echo "[eval] model=$M bench=$B  $(date +%H:%M:%S)"
    case "$B" in
      bigtom)
        MODEL="$M" PORT="$PORT" PY="$PY" bash "$REPO/scripts/run_bigtom.sh" \
          > "$REPO/logs/${M}_bigtom.log" 2>&1 ;;
      personamem)
        MODEL="$M" PORT="$PORT" PY="$PY" WORKERS=16 SIZE=32k EVAL_MODE=mcq \
          bash "$REPO/scripts/run_personamem.sh" > "$REPO/logs/${M}_personamem.log" 2>&1 ;;
      prefeval)
        MODEL="$M" PORT="$PORT" PY="$PY" bash "$REPO/scripts/run_prefeval.sh" \
          > "$REPO/logs/${M}_prefeval.log" 2>&1 ;;
    esac
    echo "[eval] model=$M bench=$B exit=$?  $(date +%H:%M:%S)"
  done
done

echo "[done] results under $REPO/results/"
