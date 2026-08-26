#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Two-H100 comparison entry point.
#
# Keep the training/runtime parameters aligned with AFH's original GiGPO
# ALFWorld recipe:
#   AFH/examples/gigpo_trainer/run_alfworld.sh
# Everything else (absolute HPC paths, offline execution, Exact Batch-ERV,
# packed roots, selected-worker branch execution, metadata and trace capture)
# is inherited from the same unified launcher used by the four-H100 run.
export GPU_COUNT=2
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export BACE_TOTAL_EPOCHS=150
export ENV_MAX_STEPS=50
export MAX_RESPONSE_LENGTH=512
export PPO_MINI_BATCH=256
# Keep 32 as the original GiGPO default. A Slurm profile may explicitly lower
# these execution-only chunk sizes after a confirmed CUDA OOM without changing
# the global train batch or PPO mini-batch.
export ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-32}
export LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-32}
export ROLLOUT_TP=2
export GPU_MEMORY_UTILIZATION=0.60
export SAVE_FREQ=-1
export TEST_FREQ=5
export VAL_BEFORE_TRAIN=true
export BACE_RESUME_MODE=disable

# Do not share artifacts with the four-GPU formal chain. Callers may still
# provide BACE_RUN_NAME explicitly for a named comparison run.
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_2gpu_gigpo_seed0}

exec bash "${SCRIPT_DIR}/run_alfworld_h100.sh" "$@"
