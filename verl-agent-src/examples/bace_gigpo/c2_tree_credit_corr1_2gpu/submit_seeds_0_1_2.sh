#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for seed in 0 1 2; do
  sbatch --parsable --partition=c23g --account=rwth2082 --time=23:30:00 --job-name="c2_corr1_s${seed}" --export="ALL,BACE_SEED=${seed},BACE_RUN_NAME=bace_c2_corr1_seed${seed}_20260908,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2" "${DIR}/run_c2_corr1_2gpu.sh"
done
