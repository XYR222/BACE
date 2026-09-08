#!/usr/bin/env bash
#SBATCH --job-name=bace_new_credit_smoke
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

MODE=${BACE_SMOKE_CREDIT_MODE:-c0_5_origin_family_local_mean}
case "${MODE}" in
    c0_5_origin_family_local_mean|c4_macro_strict_ancestor|c7_flat_leaf_gigpo|c8_macro_local_strict_ancestor) ;;
    *) echo "Unsupported BACE_SMOKE_CREDIT_MODE=${MODE}" >&2; exit 2 ;;
esac

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Concurrent smokes can share a node.  Keep Hugging Face's advisory-lock files
# node-local and job-private; the parquet inputs and model remain the canonical
# read-only assets on HPCWORK.
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/bace_hf_datasets_${SLURM_JOB_ID}"
mkdir -p "${HF_DATASETS_CACHE}"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export TRAIN_DATA_SIZE=2
export VAL_DATA_SIZE=2
export ENV_MAX_STEPS=12
export MAX_RESPONSE_LENGTH=256
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=16
export GPU_MEMORY_UTILIZATION=0.40
export SAVE_FREQ=-1
export TEST_FREQ=-1
export BACE_RESUME_MODE=disable
export SEGMENT_END_STEP=1
export LOGGER_BACKENDS="['console']"
export BACE_SEED=0

RUN_NAME="tree_credit_${MODE}_smoke_seed0_${SLURM_JOB_ID}"
INVOCATION="${SLURM_JOB_ID}_${MODE}"
export BACE_RUN_NAME=${RUN_NAME}
export BACE_INVOCATION_ID=${INVOCATION}
export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${RUN_NAME}"
export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${RUN_NAME}"
export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${RUN_NAME}"
export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${RUN_NAME}"
export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION}"
export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${RUN_NAME}/${INVOCATION}.json"
export LOG_FILE="${BACE_EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION}.log"

bash "${SCRIPT_DIR}/run_alfworld_h100_1gpu_debug.sh" \
    env.seed=0 \
    algorithm.bace.branch_pool_mode=main_reuse \
    algorithm.bace.root_active_executor=true \
    algorithm.bace.history_initial_mean=0.60 \
    algorithm.bace.credit_mode="${MODE}" \
    algorithm.bace.tree_credit_mode=current \
    algorithm.bace.macro_normalization_mode=stable_occurrence \
    algorithm.bace.tree_ppo_padding_mode=copy_trainable

"${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/validate_new_tree_credit_smoke.py" \
    "${BACE_ARTIFACT_DIR}" --mode "${MODE}" \
    --output "${BACE_EXP_ROOT}/trace_validation/${RUN_NAME}/${INVOCATION}_credit.json"
