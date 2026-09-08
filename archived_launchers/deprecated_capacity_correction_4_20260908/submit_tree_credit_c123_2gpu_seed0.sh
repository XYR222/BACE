#!/usr/bin/env bash
set -euo pipefail

# Submit three controlled full runs.  Every non-tree-credit setting is inherited
# from the already validated optimized 2-H100 BACE entry point; only the mode
# and run identity differ.  Invoke this only after the C1/C2/C3 smoke passes.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=${SCRIPT_DIR}/run_alfworld_h100_2gpu_reference_aligned.sbatch

submit_mode() {
    local label=$1
    local mode=$2
    local run_name="bace_alfworld_qwen2_5_1_5b_exact_2gpu_s3_${label}_seed0_20260902"
    local export_spec="ALL,TREE_CREDIT_MODE=${mode},MACRO_NORMALIZATION_MODE=stable_occurrence,BACE_RUN_NAME=${run_name},BACE_SEED=0,BACE_SCHEDULING_PROFILE=optimized,CAPACITY_CORRECTION_BATCH_SIZE=4,MILESTONE_CHECKPOINT_STEPS=none,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2"
    local primary
    primary=$(sbatch --parsable \
        --job-name="bace_${label}_s0" \
        --export="${export_spec}" \
        "${ENTRY}")
    local recovery
    recovery=$(sbatch --parsable \
        --dependency="afterany:${primary}" \
        --job-name="bace_${label}_s0_recover" \
        --export="${export_spec}" \
        "${ENTRY}")
    printf '%s primary=%s recovery=%s run=%s mode=%s\n' \
        "${label}" "${primary}" "${recovery}" "${run_name}" "${mode}"
}

submit_mode c1 o1_local
submit_mode c2 o1_tree_macro
submit_mode c3 o1_full_tree
