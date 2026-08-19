#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Formal 150-epoch WebShop GiGPO run. PPO micro-batch 2 passed one 10-epoch
# sample but OOMed on a heavier first batch; use 1 for worst-batch stability.
export N_GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ROLLOUT_TP=1
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}

export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export ENV_MAX_STEPS=15
export TOTAL_EPOCHS=150
export TEST_FREQ=10
export SAVE_FREQ=30
export VAL_BEFORE_TRAIN=false

export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8

export WEBSHOP_SESSIONS_PER_ACTOR=8
export WEBSHOP_SEARCH_POOL_SIZE=16
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_tp1_mem05_spa8_logprob8_ppo1_150epoch_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

printf '%s\n' \
    'WebShop GiGPO formal 150-epoch run:' \
    '  rollout tensor parallel size: 1' \
    "  vLLM GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
    '  PPO mini/micro batch: 64/1' \
    '  rollout/ref log-prob micro-batch per GPU: 8' \
    '  train/validation size: 16/128' \
    '  group size/max environment steps: 8/15' \
    '  validation/checkpoint frequency: 10/30' \
    '  validation before training: false' \
    '  sessions per actor/shared search actors: 8/16'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" "$@"
