#!/usr/bin/env bash
# Submit exactly one independent full WebShop C8/correction=1 run.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sbatch --parsable --partition=c25g --account=rwth2089 --time=15:00:00 --job-name=ws_c8_corr1_s0 \
  --export=ALL,BACE_RUN_NAME=bace_webshop1k_c8_corr1_2gpu_seed0_20260915,BACE_TOTAL_STEPS=150,BACE_SAVE_FREQ=5,BACE_MAX_CHECKPOINTS=2 \
  "${DIR}/run_webshop_c8_corr1_2gpu.sbatch"
