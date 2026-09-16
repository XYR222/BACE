#!/usr/bin/env bash
# Submit exactly the 4x3x3=36 C0 grid jobs.  This file does not run on its own.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=${SCRIPT_DIR}/run_c0_optimized_2gpu.sh
RUN_TAG=${GRID_RUN_TAG:-20260916}
for competence in 0.4 0.5 0.6 0.7; do
  for step_weight in 1.0 0.8 1.2; do
    for erv in 0.005 0.0025 0.0075; do
      slug="c${competence/./p}_w${step_weight/./p}_tau${erv/./p}"
      run_name="bace_c0_opt_seed0_${slug}_${RUN_TAG}"
      job_id=$(sbatch --parsable --partition="${SLURM_PARTITION:-c23g}" --account="${SLURM_ACCOUNT:-rwth2082}" --time="${SLURM_TIME:-23:30:00}" --job-name="c0_${slug}" \
        --export="ALL,BACE_SEED=0,C0_COMPETENCE_THRESHOLD=${competence},C0_STEP_ADVANTAGE_W=${step_weight},C0_BATCH_ERV_THRESHOLD=${erv},BACE_RUN_NAME=${run_name},TARGET_STEP=150,SAVE_FREQ=-1,MAX_CHECKPOINTS=2,MILESTONE_CHECKPOINT_STEPS=none" \
        "${ENTRY}")
      printf '%s %s\n' "${job_id}" "${run_name}"
    done
  done
done
