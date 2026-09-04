#!/usr/bin/env bash
#SBATCH --job-name=bace_c123_smoke
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

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"
# `run_alfworld_h100.sh` may invoke nested shell helpers.  Keep the repository
# importable for the post-training validator regardless of their working dir.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

run_mode() {
    local mode=$1
    local run_name="tree_credit_${mode}_smoke_seed0_${SLURM_JOB_ID}"
    local invocation="${SLURM_JOB_ID}_${mode}"
    export BACE_RUN_NAME=${run_name}
    export BACE_INVOCATION_ID=${invocation}
    export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${run_name}"
    export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${run_name}"
    export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${run_name}"
    export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${run_name}"
    export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${run_name}/${invocation}"
    export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${run_name}/${invocation}.json"
    export LOG_FILE="${BACE_EXP_ROOT}/logs/${run_name}/${invocation}.log"

    bash "${SCRIPT_DIR}/run_alfworld_h100_1gpu_debug.sh" \
        env.seed=0 \
        algorithm.bace.branch_pool_mode=main_reuse \
        algorithm.bace.root_active_executor=true \
        algorithm.bace.history_initial_mean=0.60 \
        algorithm.bace.tree_credit_mode="${mode}" \
        algorithm.bace.macro_normalization_mode=stable_occurrence \
        algorithm.bace.tree_ppo_padding_mode=copy_trainable

    "${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/validate_tree_credit_smoke.py" \
        "${BACE_ARTIFACT_DIR}" --mode "${mode}" \
        --output "${BACE_EXP_ROOT}/trace_validation/${run_name}/${invocation}_tree.json"
}

if [[ -n "${ONLY_TREE_CREDIT_MODE:-}" ]]; then
    [[ "${ONLY_TREE_CREDIT_MODE}" =~ ^(o1_local|o1_tree_macro|o1_full_tree)$ ]]
    run_mode "${ONLY_TREE_CREDIT_MODE}"
else
    run_mode o1_local
    run_mode o1_tree_macro
    run_mode o1_full_tree
fi
