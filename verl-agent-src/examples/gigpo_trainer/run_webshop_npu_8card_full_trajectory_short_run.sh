#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export N_GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# GiGPO WebShop concurrency with the selected packed-environment topology.
# train: 16 * 8 / 8 = 16 physical actors
# val:   128 / 8    = 16 physical actors
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export WEBSHOP_SESSIONS_PER_ACTOR=8
export WEBSHOP_SEARCH_POOL_SIZE=16
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

# Exercise complete WebShop trajectories while limiting training to one epoch.
export TOTAL_EPOCHS=1
export ENV_MAX_STEPS=15
export TEST_FREQ=1
export SAVE_FREQ=1
export ROLLOUT_TP=${ROLLOUT_TP:-2}

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_fulltraj_spa8_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

printf '%s\n' \
    'WebShop full-trajectory short run:' \
    '  max environment steps: 15' \
    '  total epochs: 1' \
    '  train logical sessions: 128 -> physical actors: 16' \
    '  val logical sessions:   128 -> physical actors: 16' \
    '  total physical environment actors: 32' \
    '  shared search actors: 16'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" "$@"
