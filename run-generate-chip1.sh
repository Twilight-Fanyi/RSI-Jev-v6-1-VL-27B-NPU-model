#!/bin/bash
export TRITON_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.2.1/tools/bisheng_compiler/bin/bisheng
export PATH=/usr/local/Ascend/cann-9.2.1/tools/bisheng_compiler/bin:${PATH}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.1
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.2.1
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.1/opp
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann-9.2.1/lib64:/usr/local/Ascend/cann-9.2.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-9.2.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
export PYTHONPATH=/workspace/jev-adaptation/model-44/vllm-base-probe:/usr/local/Ascend/cann-9.2.1/python/site-packages:/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec timeout 900 /usr/local/python3.12.16/bin/python3.12 /workspace/jev-adaptation/model-44/vllm-base-probe/probe-generate.py
