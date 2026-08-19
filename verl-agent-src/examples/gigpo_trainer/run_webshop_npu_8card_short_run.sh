#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Keep the full GiGPO rollout/validation concurrency while shortening trajectory
# length and training duration. This exercises all eight NPUs and output paths.
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
export GROUP_SIZE=${GROUP_SIZE:-8}

export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-2}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}

export ROLLOUT_TP=${ROLLOUT_TP:-2}
export WEBSHOP_SEARCH_POOL_SIZE=${WEBSHOP_SEARCH_POOL_SIZE:-16}
export WEBSHOP_SESSIONS_PER_ACTOR=${WEBSHOP_SESSIONS_PER_ACTOR:-8}
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_short_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" "$@"
