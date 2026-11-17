#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "$0")/weights" && pwd)}"; HOST="${VLLM_HOST:-127.0.0.1}"; PORT="${VLLM_PORT:-18044}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
while [[ $# -gt 0 ]]; do case "$1" in --model-dir) MODEL_DIR="$2";shift 2;; --output-dir) OUTPUT_DIR="$2";shift 2;; --host) HOST="$2";shift 2;; --port) PORT="$2";shift 2;; --max-model-len) MAX_MODEL_LEN="$2";shift 2;; *) echo "unknown argument: $1" >&2;exit 2;; esac; done
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$0")/logs}"; mkdir -p "$OUTPUT_DIR"
source /usr/local/Ascend/cann-9.2.1/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=ascend VLLM_WORKER_MULTIPROC_METHOD=spawn
exec /usr/local/python3.12.16/bin/vllm serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --tensor-parallel-size 2 --enforce-eager --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}"
