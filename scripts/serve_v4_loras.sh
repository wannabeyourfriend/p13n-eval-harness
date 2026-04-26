#!/bin/bash
# Serve all v4 LoRA candidates on a single base via vLLM multi-LoRA on remote GPU.
# Models: base + v4_conv_4ctors + v4_conv_plus_qa + v4_conv70_qa30
# All share the Qwen2.5-7B-Instruct base.
# Run on remote.
set -uo pipefail

PORT="${PORT:-8600}"
GPU="${GPU:-3}"
BASE_MODEL=/home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct
OUT_DIR=/home/2025user/zhou/klab-workspace/model-trainer-deployer/outputs

# LoRA paths (must exist before launch)
LORA_CONV=$OUT_DIR/qwen25_7b_instruct_v4_conv_4ctors_r64_lr2e-4/final
LORA_MIX=$OUT_DIR/qwen25_7b_instruct_v4_conv_plus_qa_r64_lr2e-4/final
LORA_7030=$OUT_DIR/qwen25_7b_instruct_v4_conv70_qa30_r64_lr2e-4/final

LORA_ARGS=()
[ -d "$LORA_CONV" ]  && LORA_ARGS+=("v4-conv-r64=$LORA_CONV")
[ -d "$LORA_MIX" ]   && LORA_ARGS+=("v4-mix-r64=$LORA_MIX")
[ -d "$LORA_7030" ]  && LORA_ARGS+=("v4-7030-r64=$LORA_7030")

if [ "${#LORA_ARGS[@]}" -eq 0 ]; then
  echo "no v4 LoRAs found yet under $OUT_DIR/qwen25_7b_instruct_v4_*" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="/home/2025user/zhou/anaconda3/envs/persona/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[serve_v4] port=$PORT gpu=$GPU loras=${LORA_ARGS[*]}"
exec /home/2025user/zhou/anaconda3/envs/persona/bin/vllm serve "$BASE_MODEL" \
  --served-model-name qwen25-7b-instruct-base \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "${LORA_ARGS[@]}" \
  --max-model-len 32768 \
  --max-lora-rank 64 \
  --max-loras "${#LORA_ARGS[@]}" \
  --gpu-memory-utilization 0.85 \
  --chat-template-content-format string
