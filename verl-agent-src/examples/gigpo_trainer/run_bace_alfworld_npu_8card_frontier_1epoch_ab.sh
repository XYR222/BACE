#!/usr/bin/env bash
set -euo pipefail

# One-epoch A/B arm for frontier batching.  Run once with
# BACE_COALESCING_ENABLED=false and once with true, keeping all NPU settings
# identical.  The underlying 8-card script remains the single source of
# memory/runtime configuration.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export AESC_RUN_NAME=${AESC_RUN_NAME:-bace_frontier_alfworld_npu_8card_1epoch_ab_coalescing}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
export GROUP_SIZE=${GROUP_SIZE:-8}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}

COALESCING_ENABLED=${BACE_COALESCING_ENABLED:-true}
COALESCING_MAX_BATCH_SIZE=${BACE_COALESCING_MAX_BATCH_SIZE:-0}
COALESCING_MIN_BATCH_SIZE=${BACE_COALESCING_MIN_BATCH_SIZE:-32}

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_frontier.sh" \
    algorithm.bace.frontier_batch_coalescing.enabled="${COALESCING_ENABLED}" \
    algorithm.bace.frontier_batch_coalescing.max_batch_size="${COALESCING_MAX_BATCH_SIZE}" \
    algorithm.bace.frontier_batch_coalescing.min_batch_size="${COALESCING_MIN_BATCH_SIZE}" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    "$@"
