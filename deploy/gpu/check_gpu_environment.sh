#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_PATH=${MODEL_PATH:-${BUNDLE_ROOT}/assets/models/Qwen2.5-1.5B-Instruct}
ALFWORLD_DATA=${ALFWORLD_DATA:-${BUNDLE_ROOT}/assets/alfworld}

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found" >&2; exit 1; }
nvidia-smi

if [[ -z "${GPU_COUNT:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES}"
        GPU_COUNT=${#ids[@]}
    else
        GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    fi
fi

python3 "${SCRIPT_DIR}/check_gpu_environment.py" \
    --expected-gpus "${GPU_COUNT}" --model-path "${MODEL_PATH}" --alfworld-data "${ALFWORLD_DATA}"
