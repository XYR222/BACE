#!/usr/bin/env bash
set -euo pipefail

# Real 8-NPU, one-update smoke for the BatchERV Exact main method.
# It inherits the validated GiGPO-derived runtime/memory settings and writes
# all rollout and BACE diagnostic artifacts under AESC_EXP_ROOT.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export AESC_EXP_ROOT=${AESC_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
export AESC_RUN_NAME=${AESC_RUN_NAME:-bace_alfworld_npu_8card_1epoch_batch_erv_exact_smoke}
export TOTAL_EPOCHS=1
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
export GROUP_SIZE=${GROUP_SIZE:-8}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_150epoch_batch_erv_exact.sh" \
    algorithm.bace.invalid_action_mode=strict_identity \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    "$@"
