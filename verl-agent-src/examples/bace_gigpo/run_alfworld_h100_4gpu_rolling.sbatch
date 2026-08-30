#!/usr/bin/env bash
#SBATCH --job-name=bace15b_main_roll
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=23:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

TARGET_STEP=${TARGET_STEP:-150}
ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-16}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-2}
SAVE_FREQ=${SAVE_FREQ:-5}
MILESTONE_CHECKPOINT_STEPS=${MILESTONE_CHECKPOINT_STEPS:-10,75,140}
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP >= 1 && TARGET_STEP <= 150 ))
[[ "${ACTOR_MICRO_BATCH}" =~ ^(32|16|8)$ ]]
[[ "${MAX_CHECKPOINTS}" =~ ^(1|2)$ ]]
[[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]] && (( SAVE_FREQ >= 1 ))

export BACE_EXP_ROOT=${BACE_EXP_ROOT:-/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact}
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_main_seed0}
export BACE_INVOCATION_ID=${SLURM_JOB_ID:-local}_rolling_to_step_${TARGET_STEP}
export BACE_RESUME_MODE=auto
export SEGMENT_END_STEP=${TARGET_STEP}
export SAVE_FREQ
export TEST_FREQ=5
export ACTOR_MICRO_BATCH
export LOGPROB_MICRO_BATCH=${ACTOR_MICRO_BATCH}
export MAX_CHECKPOINTS
export MILESTONE_CHECKPOINT_STEPS
unset CHECKPOINT_DIR BACE_ARTIFACT_DIR ROLLOUT_DATA_DIR TENSORBOARD_DIR
unset RUN_METADATA_DIR TRACE_VALIDATION_REPORT LOG_FILE

checkpoint_root=${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}
export MILESTONE_CHECKPOINT_DIR=${BACE_EXP_ROOT}/preserved_checkpoints/${BACE_RUN_NAME}
artifact_root=${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}
metadata_dir=${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}
report_dir=${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}
tracker=${checkpoint_root}/latest_checkpointed_iteration.txt
mkdir -p "${checkpoint_root}" "${metadata_dir}" "${report_dir}"

if [[ "${DRY_RUN:-0}" != 1 && -f "${tracker}" ]]; then
    python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${checkpoint_root}" \
        --keep "${MAX_CHECKPOINTS}" \
        --output "${metadata_dir}/checkpoint_prune_before.json"
    python3 -m recipe.bace_gigpo.validate_trace "${artifact_root}" \
        --output "${report_dir}/${BACE_INVOCATION_ID}_before_resume.json"
    latest_step=$(<"${tracker}")
    if (( latest_step >= TARGET_STEP )); then
        printf 'Target step %s already checkpointed and validated; nothing to resume.\n' \
            "${TARGET_STEP}"
        exit 0
    fi
fi

# Unlike the retained legacy segment launcher, this invocation always targets
# the final step.  A replacement wall-clock allocation resumes whichever
# periodic checkpoint was last committed; there are no 20/40/... boundaries.
bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh" \
    algorithm.bace.branch_pool_mode=main_reuse \
    algorithm.bace.root_active_executor=true

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

[[ -f "${tracker}" ]]
[[ "$(<"${tracker}")" == "${TARGET_STEP}" ]]
python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${checkpoint_root}" \
    --keep "${MAX_CHECKPOINTS}" \
    --output "${metadata_dir}/checkpoint_prune_after.json"
python3 -m recipe.bace_gigpo.validate_trace "${artifact_root}" \
    --output "${report_dir}/${BACE_INVOCATION_ID}_final.json"
