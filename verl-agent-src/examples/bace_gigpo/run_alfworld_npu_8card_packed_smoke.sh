#!/usr/bin/env bash
set -euo pipefail

# One-epoch correctness and memory smoke for the full 16-task x 8-slot
# BACE staged-packed pipeline. The underlying script owns all training,
# artifact, checkpoint, trace-validation, and platform-argument handling.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_packed_alfworld_npu_8card_1epoch_smoke_128slots}
export BACE_TOTAL_EPOCHS=${BACE_TOTAL_EPOCHS:-1}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
export BACE_TOTAL_LEAF_BUDGET=${BACE_TOTAL_LEAF_BUDGET:-8}
export BACE_STAGED_ROOT_BATCHING=packed
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}
# Match the known-good frontier smoke memory configuration for a fair
# packed-vs-frontier comparison. The 50-epoch entrypoint remains conservative
# by default and can still be overridden independently.
export BACE_ACTOR_MICRO_BATCH_SIZE_PER_GPU=${BACE_ACTOR_MICRO_BATCH_SIZE_PER_GPU:-8}
export BACE_GPU_MEMORY_UTILIZATION=${BACE_GPU_MEMORY_UTILIZATION:-0.6}

exec bash "${SCRIPT_DIR}/run_alfworld_npu_8card_50epoch.sh" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    "$@"
