#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export N_GPUS_PER_NODE=2
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

# Keep the original GiGPO WebShop task/batch protocol.
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export WEBSHOP_SESSIONS_PER_ACTOR=8
export WEBSHOP_SEARCH_POOL_SIZE=16
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

# Original WebShop trajectory length, but one epoch for a controlled smoke run.
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-15}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}

# First controlled adaptation from upstream: preserve the PPO update and vLLM
# budget, while halving only the log-prob micro-batches that triggered OOM.
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
export ROLLOUT_TP=2

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_2card_short_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

printf '%s\n' \
    'WebShop NPU two-card GiGPO short run:' \
    '  max environment steps: 15' \
    '  total epochs: 1' \
    '  train/val logical sessions: 128/128' \
    '  sessions per physical actor: 8' \
    '  total physical environment actors: 32' \
    '  shared search actors: 16' \
    '  PPO micro-batch per GPU: 8' \
    '  rollout/ref log-prob micro-batch per GPU: 8' \
    '  vLLM GPU memory utilization: 0.6'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    "$@"
