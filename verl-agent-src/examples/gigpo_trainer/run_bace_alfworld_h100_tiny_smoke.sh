#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUNDLE_ROOT=$(cd "${REPO_ROOT}/.." && pwd)
WORKSPACE_ROOT=$(cd "${BUNDLE_ROOT}/.." && pwd)

export GPU_COUNT=1
export MODEL_PATH=${MODEL_PATH:-${WORKSPACE_ROOT}/model_down/model/Qwen2.5-3B-Instruct}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${HOME}/.cache/alfworld}
export TRAIN_FILE=${TRAIN_FILE:-${WORKSPACE_ROOT}/data/text/train.parquet}
export VAL_FILE=${VAL_FILE:-${WORKSPACE_ROOT}/data/text/test.parquet}
export BACE_EXP_ROOT=${BACE_EXP_ROOT:-${BUNDLE_ROOT}/experiments/h100-tiny-smoke}
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_h100_tiny_smoke_$(date -u +%Y%m%dT%H%M%SZ)}

export BACE_TOTAL_EPOCHS=1
export TRAIN_DATA_SIZE=1
export VAL_DATA_SIZE=1
export GROUP_SIZE=2
export ENV_MAX_STEPS=2
export MAX_RESPONSE_LENGTH=64
export ACTOR_MICRO_BATCH=1
export LOGPROB_MICRO_BATCH=1
export PPO_MINI_BATCH=2
export GPU_MEMORY_UTILIZATION=0.25
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=8
export ROLLOUT_TP=1
export SAVE_FREQ=-1
export TEST_FREQ=-1
export NUM_CPUS_PER_ENV_WORKER=0.1
export LOGGER_BACKENDS="['console']"
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

# vLLM 0.11 uses its CUDA memory pool for colocated rollout weights. PyTorch's
# expandable allocator is incompatible with that pool, including when inherited.
unset PYTORCH_CUDA_ALLOC_CONF

exec bash "${SCRIPT_DIR}/run_bace_alfworld_gpu.sh" \
    algorithm.bace.total_leaf_budget=2 \
    algorithm.bace.min_natural_roots=1 \
    algorithm.bace.max_branches_per_anchor=1 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    trainer.total_training_steps=1 \
    "$@"
