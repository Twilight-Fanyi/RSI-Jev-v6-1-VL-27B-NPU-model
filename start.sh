#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=""
OUTPUT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
: "${MODEL_DIR:?--model-dir is required}"
: "${OUTPUT_DIR:?--output-dir is required}"
export ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.1
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.2.1
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.1/opp
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann-9.2.1/lib64:/usr/local/Ascend/cann-9.2.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.2.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:
export PYTHONPATH="$SCRIPT_DIR/vllm-base-probe:/usr/local/Ascend/cann-9.2.1/python/site-packages:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export TRITON_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.2.1/tools/bisheng_compiler/bin/bisheng
export PATH=/usr/local/Ascend/cann-9.2.1/tools/bisheng_compiler/bin:${PATH}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=ascend VLLM_WORKER_MULTIPROC_METHOD=spawn
exec /usr/local/python3.12.16/bin/python3.12 "$SCRIPT_DIR/inference.py" --serve --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-18143}"
