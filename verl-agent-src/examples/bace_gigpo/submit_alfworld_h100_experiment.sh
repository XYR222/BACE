#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
MODEL_PATH=/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct

for path in \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/tokenizer_config.json" \
    /hpcwork/xsz96350/fu_project/data/text/train.parquet \
    /hpcwork/xsz96350/fu_project/data/text/test.parquet \
    /home/xsz96350/.cache/alfworld; do
    [[ -e "${path}" ]] || { echo "Required asset missing: ${path}" >&2; exit 2; }
done

mkdir -p "${FORMAL_ROOT}/slurm" "${FORMAL_ROOT}/chain"
debug_job=$(sbatch --parsable "${SCRIPT_DIR}/slurm_alfworld_h100_1gpu_debug.sbatch")
profile_job=$(sbatch --parsable --dependency="afterok:${debug_job}" \
    "${SCRIPT_DIR}/slurm_alfworld_h100_4gpu_profile.sbatch")

submission=${FORMAL_ROOT}/chain/initial_submission_$(date -u +%Y%m%dT%H%M%SZ).txt
printf 'debug_job_id=%s\nprofile_job_id=%s\nprofile_dependency=afterok:%s\n' \
    "${debug_job}" "${profile_job}" "${debug_job}" | tee "${submission}"
echo "The profile job will submit the step-150 afterok chain after recovery validation."
