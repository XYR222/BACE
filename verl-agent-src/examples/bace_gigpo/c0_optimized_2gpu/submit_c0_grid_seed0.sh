#!/usr/bin/env bash
# Submit exactly the 3x3x3=27 C0 grid jobs.  This file does not run on its own.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRY=${SCRIPT_DIR}/run_c0_optimized_2gpu.sh
for competence in 0.5 0.6 0.7; do
  for step_weight in 1.0 0.8 1.2; do
    for erv in 0.005 0.0025 0.0075; do
      slug="c${competence/./p}_w${step_weight/./p}_tau${erv/./p}"
      sbatch --parsable --partition="${SLURM_PARTITION:-c23g}" --account="${SLURM_ACCOUNT:-rwth2082}" --time="${SLURM_TIME:-23:30:00}" --job-name="c0_${slug}" \
        --export="ALL,BACE_SEED=0,C0_COMPETENCE_THRESHOLD=${competence},C0_STEP_ADVANTAGE_W=${step_weight},C0_BATCH_ERV_THRESHOLD=${erv},BACE_RUN_NAME=bace_c0_opt_seed0_${slug}_20260908,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2" \
        "${ENTRY}"
    done
  done
done
