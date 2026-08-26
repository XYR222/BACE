#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUNDLE_ROOT=$(cd "${REPO_ROOT}/.." && pwd)
WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project

GPU_COUNT=${GPU_COUNT:?GPU_COUNT must be set to 1, 2, or 4}
case "${GPU_COUNT}" in 1|2|4) ;; *) echo "GPU_COUNT must be 1, 2, or 4" >&2; exit 2 ;; esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
fi

export MODEL_PATH=${MODEL_PATH:-${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/xsz96350/.cache/alfworld}
export TRAIN_FILE=${TRAIN_FILE:-${WORKSPACE_ROOT}/data/text/train.parquet}
export VAL_FILE=${VAL_FILE:-${WORKSPACE_ROOT}/data/text/test.parquet}
export BACE_EXP_ROOT=${BACE_EXP_ROOT:-${BUNDLE_ROOT}/experiments/alfworld-qwen2.5-1.5b-exact}
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_main_seed0}
invocation_default=${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)
export BACE_INVOCATION_ID=${BACE_INVOCATION_ID:-${invocation_default}}

export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}}
export BACE_ARTIFACT_DIR=${BACE_ARTIFACT_DIR:-${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}}
export RUN_METADATA_DIR=${RUN_METADATA_DIR:-${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}}
export TRACE_VALIDATION_REPORT=${TRACE_VALIDATION_REPORT:-${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.json}
export LOG_FILE=${LOG_FILE:-${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log}

export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
export GROUP_SIZE=${GROUP_SIZE:-8}
export BACE_TOTAL_EPOCHS=${BACE_TOTAL_EPOCHS:-150}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-50}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export PPO_MINI_BATCH=${PPO_MINI_BATCH:-256}
export ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-32}
export LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-32}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
export BACE_RESUME_MODE=${BACE_RESUME_MODE:-auto}
export LOGGER_BACKENDS=${LOGGER_BACKENDS:-"['console','tensorboard']"}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# Direct interactive invocations do not necessarily source h100_job_env.sh.
# Apply the same bounded CPU/thread policy here so every Ray actor inherits it.
export BACE_CPU_THREADS=${BACE_CPU_THREADS:-1}
export OMP_NUM_THREADS=${BACE_CPU_THREADS}
export OMP_THREAD_LIMIT=${BACE_CPU_THREADS}
export MKL_NUM_THREADS=${BACE_CPU_THREADS}
export OPENBLAS_NUM_THREADS=${BACE_CPU_THREADS}
export NUMEXPR_NUM_THREADS=${BACE_CPU_THREADS}
export NUMEXPR_MAX_THREADS=${BACE_CPU_THREADS}
export VECLIB_MAXIMUM_THREADS=${BACE_CPU_THREADS}
export RAYON_NUM_THREADS=${BACE_CPU_THREADS}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-$((GPU_COUNT * 16))}}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${BACE_EXP_ROOT}" "${RUN_METADATA_DIR}"

overrides=(
    algorithm.bace.history_initial_mean=0.10
    algorithm.bace.history_initial_strength=2.0
    algorithm.bace.history_forgetting=0.8
    algorithm.bace.batch_erv_threshold=0.005
    algorithm.bace.total_leaf_budget=8
    algorithm.bace.min_natural_roots=2
    algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.5
    algorithm.bace.history_transfer_fraction=0.1
    algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0
    algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity
    algorithm.bace.local_credit_mode=occurrence
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.kl_loss_coef=0.01
    algorithm.gamma=0.95
    ray_init.num_cpus="${RAY_NUM_CPUS}"
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS:-2}"
    trainer.validate_on_last_step=false
)
if [[ -n "${SEGMENT_END_STEP:-}" ]]; then
    overrides+=(trainer.total_training_steps="${SEGMENT_END_STEP}")
fi

set +e
bash "${REPO_ROOT}/examples/gigpo_trainer/run_bace_alfworld_gpu.sh" \
    "${overrides[@]}" "$@"
status=$?
set -e
if [[ ${status} -ne 0 ]]; then
    exit "${status}"
fi

if [[ "${DRY_RUN:-0}" != 1 && -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep "${MAX_CHECKPOINTS:-2}" \
        --output "${RUN_METADATA_DIR}/checkpoint_prune.json"
fi
