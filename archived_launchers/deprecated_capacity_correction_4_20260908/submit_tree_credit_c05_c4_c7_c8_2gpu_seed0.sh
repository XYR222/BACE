#!/usr/bin/env bash
set -euo pipefail

# Submit the four seed-0 C0.5/C4/C7/C8 main runs only after their smoke gate
# has passed.  The entry point is self-contained and keeps all non-credit
# training parameters aligned with the existing 2-H100 reference launcher.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=${SCRIPT_DIR}/run_alfworld_h100_2gpu_reference_aligned.sbatch

submit_mode() {
    local label=$1
    local credit_mode=$2
    local run_name="bace_alfworld_qwen2_5_1_5b_exact_2gpu_${label}_seed0_20260906"
    local export_spec="ALL,BACE_CREDIT_MODE=${credit_mode},TREE_CREDIT_MODE=current,MACRO_NORMALIZATION_MODE=stable_occurrence,BACE_RUN_NAME=${run_name},BACE_SEED=0,BACE_SCHEDULING_PROFILE=optimized,CAPACITY_CORRECTION_BATCH_SIZE=4,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2,MILESTONE_CHECKPOINT_STEPS=none"
    local job_id
    # Slurm's current 2-GPU/15-hour backfill estimate is materially earlier on
    # c23g than c25g.  Override the legacy entry's directives without altering
    # any reference training parameter.
    job_id=$(sbatch --parsable \
        --partition=c23g --account=rwth2089 --time=15:00:00 \
        --job-name="bace_${label}_s0" \
        --export="${export_spec}" \
        "${ENTRY}")
    printf '%s job=%s run=%s credit_mode=%s partition=c23g time=15:00:00\n' \
        "${label}" "${job_id}" "${run_name}" "${credit_mode}"
}

submit_mode c05 c0_5_origin_family_local_mean
submit_mode c4 c4_macro_strict_ancestor
submit_mode c7 c7_flat_leaf_gigpo
submit_mode c8 c8_macro_local_strict_ancestor
