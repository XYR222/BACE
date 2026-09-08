#!/usr/bin/env bash
set -euo pipefail

# Recovery chain for the 2026-09-06 seed-0 formal runs.  Serial afterany
# dependencies prevent two large Ray/ALFWorld actor pools from being placed on
# the same four-GPU node again.  C0.5 resumes from its latest complete
# checkpoint; C4/C7 start at zero because their failed jobs never trained.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=${SCRIPT_DIR}/run_alfworld_h100_2gpu_reference_aligned.sbatch

submit_one() {
    local dependency=$1
    local label=$2
    local credit_mode=$3
    local run_name=$4
    local export_spec="ALL,BACE_CREDIT_MODE=${credit_mode},TREE_CREDIT_MODE=current,MACRO_NORMALIZATION_MODE=stable_occurrence,BACE_RUN_NAME=${run_name},BACE_SEED=0,BACE_SCHEDULING_PROFILE=optimized,CAPACITY_CORRECTION_BATCH_SIZE=4,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2,MILESTONE_CHECKPOINT_STEPS=none"
    local args=(
        --parsable
        --partition=c23g
        --account=rwth2089
        --time=15:00:00
        --job-name="bace_${label}_s0_r"
        --export="${export_spec}"
    )
    if [[ -n "${dependency}" ]]; then
        args+=(--dependency="afterany:${dependency}")
    fi
    sbatch "${args[@]}" "${ENTRY}"
}

c05_job=$(submit_one "" c05 c0_5_origin_family_local_mean \
    bace_alfworld_qwen2_5_1_5b_exact_2gpu_c05_seed0_20260906)
c4_job=$(submit_one "${c05_job}" c4 c4_macro_strict_ancestor \
    bace_alfworld_qwen2_5_1_5b_exact_2gpu_c4_seed0_20260906)
c7_job=$(submit_one "${c4_job}" c7 c7_flat_leaf_gigpo \
    bace_alfworld_qwen2_5_1_5b_exact_2gpu_c7_seed0_20260906)

printf 'C0.5 resume job=%s\nC4 restart job=%s dependency=afterany:%s\nC7 restart job=%s dependency=afterany:%s\n' \
    "${c05_job}" "${c4_job}" "${c05_job}" "${c7_job}" "${c4_job}"
