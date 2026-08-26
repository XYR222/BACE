#!/usr/bin/env bash
#SBATCH --job-name=bace15b_2g_gigpo
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=23:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_RUN_NAME=bace_alfworld_qwen2_5_1_5b_exact_2gpu_gigpo_seed0
export BACE_INVOCATION_ID=${SLURM_JOB_ID}_2gpu_gigpo_formal
# The original per-GPU value 32 was tested by gate 3082011 and failed in
# update_actor with a confirmed CUDA OOM. Keep all global GiGPO parameters
# unchanged and use the approved execution-chunk fallback 32 -> 16.
export ACTOR_MICRO_BATCH=16
export LOGPROB_MICRO_BATCH=16
unset CHECKPOINT_DIR BACE_ARTIFACT_DIR ROLLOUT_DATA_DIR TENSORBOARD_DIR
unset RUN_METADATA_DIR TRACE_VALIDATION_REPORT LOG_FILE

bash "${SCRIPT_DIR}/run_alfworld_h100_2gpu.sh"

artifact_dir=${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}
log_file=${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log
report_dir=${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}

grep -Eq "training/global_step['\"]?:[[:space:]]*150|training/global_step[^0-9]+150" \
    "${log_file}"
python -m recipe.bace_gigpo.validate_trace "${artifact_dir}" \
    --output "${report_dir}/${BACE_INVOCATION_ID}_final.json"
python "${SCRIPT_DIR}/validate_selected_worker_metrics.py" "${artifact_dir}" \
    --output "${report_dir}/${BACE_INVOCATION_ID}_selected_worker.json"
