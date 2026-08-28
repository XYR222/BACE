#!/usr/bin/env bash
#SBATCH --job-name=bace_p1s_s23
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

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

run_profile() {
    local label=$1
    local pool_mode=$2
    export BACE_RUN_NAME="p1s_${label}_${SLURM_JOB_ID}"
    export BACE_INVOCATION_ID="${SLURM_JOB_ID}_${label}"
    export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}"
    export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}"
    export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}"
    export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}"
    export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}"
    export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.json"
    export LOG_FILE="${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log"

    bash "${SCRIPT_DIR}/run_alfworld_h100_1gpu_debug.sh" \
        algorithm.bace.branch_pool_mode="${pool_mode}" \
        algorithm.bace.root_active_executor=true \
        algorithm.bace.history_initial_mean=0.60

    python -m recipe.bace_gigpo.validate_trace \
        "${BACE_ARTIFACT_DIR}" \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_final.json"
    python "${SCRIPT_DIR}/validate_p1s_profile.py" \
        "${BACE_ARTIFACT_DIR}" --pool-mode "${pool_mode}" --root-active true \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_p1s.json"
}

run_profile s2 dedicated
run_profile s3 main_reuse
