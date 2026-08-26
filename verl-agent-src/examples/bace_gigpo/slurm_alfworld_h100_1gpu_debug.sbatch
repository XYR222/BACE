#!/usr/bin/env bash
#SBATCH --job-name=bace15b_1g_debug
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_INVOCATION_ID=${SLURM_JOB_ID}_1gpu_debug
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_1gpu_debug}

debug_overrides=()
if [[ "${BACE_FORCE_BRANCH_SMOKE:-0}" == 1 ]]; then
    # This affects only the disposable debug run.  It forces a branch in the
    # first policy update so the selected-worker path is exercised on H100.
    debug_overrides+=(algorithm.bace.history_initial_mean=0.60)
fi
bash "${SCRIPT_DIR}/run_alfworld_h100_1gpu_debug.sh" "${debug_overrides[@]}"
python -m recipe.bace_gigpo.validate_trace \
    "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
    --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_final.json"
if [[ "${BACE_REQUIRE_SELECTED_BRANCH:-0}" == 1 ]]; then
    python "${SCRIPT_DIR}/validate_selected_worker_metrics.py" \
        "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_selected_worker.json"
fi
if [[ "${BACE_REQUIRE_TASK_ROTATION:-0}" == 1 ]]; then
    python "${SCRIPT_DIR}/validate_alfworld_task_rotation.py" \
        "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
        --first-step 1 --second-step 2 \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_task_rotation.json"
fi
grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" \
    "${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log"

if [[ "${BACE_SUBMIT_4GPU_AFTER_SUCCESS:-0}" == 1 ]]; then
    : "${BACE_FORMAL_RUN_NAME:?BACE_FORMAL_RUN_NAME is required for gated submission}"
    formal_checkpoint_root=${BACE_EXP_ROOT}/checkpoints/${BACE_FORMAL_RUN_NAME}
    if [[ -e "${formal_checkpoint_root}/latest_checkpointed_iteration.txt" ]]; then
        echo "Refusing step-0 submission: formal checkpoint tracker already exists" >&2
        exit 1
    fi
    bash "${SCRIPT_DIR}/submit_alfworld_h100_4gpu_rolling.sh" \
        --target-step 150 \
        --micro-batch "${BACE_FORMAL_MICRO_BATCH:-16}" \
        --max-checkpoints "${BACE_FORMAL_MAX_CHECKPOINTS:-2}" \
        --run-name "${BACE_FORMAL_RUN_NAME}" \
        --partition "${BACE_FORMAL_PARTITION:-c25g}" \
        --account "${BACE_FORMAL_ACCOUNT:-rwth2089}"
fi
