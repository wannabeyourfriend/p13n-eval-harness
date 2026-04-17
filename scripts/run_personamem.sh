#!/bin/bash
# Run PersonaMem-v2 (MCQ mode by default) for a single served model.
#
# Required env:
#   MODEL        - friendly output name
#   PORT         - local vLLM port
# Optional env:
#   MODEL_ID     - actual served id (default: $MODEL)
#   WORKERS      - concurrency (default 32)
#   SIZE         - 32k or 128k (default 32k)
#   EVAL_MODE    - mcq / generative / both (default mcq)
#   BENCH_FILE   - path to benchmark CSV (default:
#                  data/personamem/benchmark/text/benchmark.csv)
#   RUN_JUDGES   - 1 to run generative-mode LLM judges (default off)
#   OUT_ROOT     - results root
set -uo pipefail

: "${MODEL:?set MODEL}"
: "${PORT:?set PORT}"
MODEL_ID="${MODEL_ID:-$MODEL}"
WORKERS="${WORKERS:-32}"
SIZE="${SIZE:-32k}"
EVAL_MODE="${EVAL_MODE:-mcq}"
RUN_JUDGES="${RUN_JUDGES:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
OUT="$OUT_ROOT/$MODEL"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
BENCH_FILE="${BENCH_FILE:-$REPO/data/personamem/benchmark/text/benchmark.csv}"

PY="${PY:-/home/2025user/zhou/anaconda3/envs/persona/bin/python}"
API_BASE="http://localhost:${PORT}/v1"
export OPENAI_BASE_URL="$API_BASE"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
# Some upstream PersonaMem code paths import transformers (PIL chain) — preload
# conda libstdc++ to avoid GLIBCXX errors on the host.
export LD_PRELOAD="${LD_PRELOAD:-/home/2025user/zhou/anaconda3/envs/persona/lib/libstdc++.so.6}"

extra=()
[ -n "$RUN_JUDGES" ] && extra+=(--run-judges)

# Upstream PersonaMem-v2 reads chat_history_*_link as paths RELATIVE to its own
# data root (e.g. "data/chat_history_32k/..."). Run with cwd at data/personamem
# so those resolve correctly. Add repo to PYTHONPATH because cd breaks the
# default `python -m multibench.cli` import path.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"
cd "$REPO/data/personamem"
echo "[personamem] model=$MODEL port=$PORT size=$SIZE eval_mode=$EVAL_MODE workers=$WORKERS  cwd=$(pwd)"
"$PY" -m multibench.cli run personamem -- \
    --api-base "$API_BASE" --model "$MODEL_ID" --workers "$WORKERS" \
    --eval-mode "$EVAL_MODE" --size "$SIZE" \
    --benchmark-file "$BENCH_FILE" \
    --output-dir "$OUT/PersonaMem" \
    "${extra[@]}" \
    2>&1 | tee "$LOGDIR/personamem.log"
echo "[personamem] done $(date)"
