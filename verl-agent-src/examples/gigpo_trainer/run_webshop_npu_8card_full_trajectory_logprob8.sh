#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export N_GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Keep the selected SPA=8 topology and the original GiGPO global batch.
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export WEBSHOP_SESSIONS_PER_ACTOR=8
export WEBSHOP_SEARCH_POOL_SIZE=16
export WEBSHOP_SEARCH_ACTOR_CPUS=${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}
export WEBSHOP_WORKER_INIT_BATCH_SIZE=${WEBSHOP_WORKER_INIT_BATCH_SIZE:-4}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.05}

# Full WebShop trajectories, one training epoch for the diagnostic run.
export TOTAL_EPOCHS=1
export ENV_MAX_STEPS=15
export TEST_FREQ=1
export SAVE_FREQ=1

# The original value is 16. Halve only log-prob micro-batches to reduce the
# entropy/log-prob peak that caused the previous NPU OOM.
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
export ROLLOUT_TP=${ROLLOUT_TP:-2}

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_fulltraj_spa8_logprob8_${run_timestamp}}
export RESUME_MODE=${RESUME_MODE:-disable}

printf '%s\n' \
    'WebShop full-trajectory log-prob memory diagnostic:' \
    '  max environment steps: 15' \
    '  train/val logical sessions: 128/128' \
    '  sessions per physical actor: 8' \
    '  total physical environment actors: 32' \
    '  shared search actors: 16' \
    '  rollout/ref log-prob micro-batch per GPU: 8'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_train.sh" "$@"
