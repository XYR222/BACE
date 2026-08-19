#!/usr/bin/env bash
set -euo pipefail

# Coalescing arm for the 1-epoch BACE frontier A/B check.
# This enables optional ready-job batch coalescing without changing the
# underlying root/branch quota or replay identity logic.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ENGINE=vllm
export AESC_EXP_ROOT=/opt/dpcvol/datasets/8165423358032568398/AESC-exp
export AESC_RUN_NAME=bace_frontier_alfworld_npu_8card_1epoch_ab_coalescing
export TOTAL_EPOCHS=1
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=8
export GROUP_SIZE=8
export ENV_MAX_STEPS=40
export MODEL_PATH=/opt/dpcvol/datasets/8165423358032568398/model/Qwen2.5-1.5B-Instruct
export ROLLOUT_TP=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BUNDLED_CONDA_ENV=/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld
export CONDA_SH=/home/naie/Asend/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

COALESCING_ENABLED=true
COALESCING_MAX_BATCH_SIZE=0
COALESCING_MIN_BATCH_SIZE=32

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_frontier.sh" \
    algorithm.bace.frontier_batch_coalescing.enabled="${COALESCING_ENABLED}" \
    algorithm.bace.frontier_batch_coalescing.max_batch_size="${COALESCING_MAX_BATCH_SIZE}" \
    algorithm.bace.frontier_batch_coalescing.min_batch_size="${COALESCING_MIN_BATCH_SIZE}" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
