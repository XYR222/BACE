#!/usr/bin/env bash
set -euo pipefail

# One-epoch correctness smoke for the full 16-task x 8-slot BACE frontier.
# The underlying script retains the proven 8-card NPU memory configuration.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export AESC_RUN_NAME=${AESC_RUN_NAME:-bace_frontier_alfworld_npu_8card_1epoch_smoke_128slots}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
export GROUP_SIZE=${GROUP_SIZE:-8}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_frontier.sh" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    "$@"
