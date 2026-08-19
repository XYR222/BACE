#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# TP=1 stability run. Keep the proven WebShop training and packed-environment
# settings while running enough epochs to encounter a wider sequence mix.
export N_GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ROLLOUT_TP=1

export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export ENV_MAX_STEPS=15
export TOTAL_EPOCHS=10
export TEST_FREQ=5
export SAVE_FREQ=5

export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
# TP=1 at 0.6 reserved 93.918 GB and later OOMed in actor backward. Leave
# additional workspace headroom while retaining 0.6 as the base-script default.
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}

export WEBSHOP_SESSIONS_PER_ACTOR=8
export WEBSHOP_SEARCH_POOL_SIZE=16
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_tp1_mem05_spa8_logprob8_ppo2_10epoch_stability_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

printf '%s\n' \
    'WebShop eight-card TP=1 stability run:' \
    '  epochs: 10' \
    '  validation/checkpoint frequency: 5' \
    '  rollout tensor parallel size: 1' \
    '  rollout replicas: 8' \
    "  vLLM GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}" \
    '  PPO mini/micro batch: 64/2' \
    '  rollout/ref log-prob micro-batch per GPU: 8' \
    '  train/validation size: 16/128' \
    '  group size/max environment steps: 8/15' \
    '  sessions per actor/shared search actors: 8/16'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" "$@"
