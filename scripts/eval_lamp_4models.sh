#!/bin/bash
# Launch LaMP-1..7 + LaMP-QA for all 4 served models in parallel.
#
# Ports:
#   8001  no-state-us
#   8002  us-profile-mar31
#   8003  oracle-profile-only
#   8004  Qwen2.5-7B-Instruct (base + default LaMP-QA judge)
#
# Each model hits its own endpoint for LaMP generation and (by default) uses
# port 8004 as the shared judge so LoRA runs are comparable.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-32}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JUDGE_PORT="${JUDGE_PORT:-8004}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct}"

wait_for_port() {
    local port="$1" model="$2"
    local tries=0
    until curl -sS -m 3 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "data"; do
        tries=$((tries+1))
        if (( tries == 1 )); then echo "[$model] waiting for port $port to come up..."; fi
        if (( tries > 120 )); then  # 120 * 15s = 30min max
            echo "[$model] gave up waiting for port $port"; return 1
        fi
        sleep 15
    done
}

run_one() {
    local model="$1" port="$2" model_id="$3"
    wait_for_port "$port" "$model" || return 1
    MODEL="$model" PORT="$port" MODEL_ID="$model_id" WORKERS="$WORKERS" \
        OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/run_lamp.sh" \
        > "$OUT_ROOT/_${model}_lamp.log" 2>&1
    wait_for_port "$JUDGE_PORT" "$model(judge)" || return 1
    MODEL="$model" PORT="$port" MODEL_ID="$model_id" WORKERS="$WORKERS" \
        JUDGE_PORT="$JUDGE_PORT" JUDGE_MODEL="$JUDGE_MODEL" \
        OUT_ROOT="$OUT_ROOT" \
        bash "$REPO/scripts/run_lampqa.sh" \
        > "$OUT_ROOT/_${model}_lampqa.log" 2>&1
    echo "[done] $model"
}

echo "Launching 4 models in parallel (LaMP + LaMP-QA)"
echo "  judge = $JUDGE_MODEL @ :$JUDGE_PORT"

run_one no-state-us          8001 "no-state-us"          &
run_one us-profile-mar31     8002 "us-profile-mar31"     &
run_one oracle-profile-only  8003 "oracle-profile-only"  &
run_one qwen2.5-7b-instruct  8004 "/home/2025user/zhou/hf_models/Qwen2.5-7B-Instruct" &

wait
echo "=== all LaMP + LaMP-QA done $(date) ==="
