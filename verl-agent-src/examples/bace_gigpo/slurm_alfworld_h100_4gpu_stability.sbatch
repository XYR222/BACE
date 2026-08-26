#!/usr/bin/env bash
#SBATCH --job-name=bace15b_stability
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
STABILITY_ROOT=${FORMAL_ROOT}/stability_profile
stability_micro_batch=${STABILITY_MICRO_BATCH:-16}
formal_max_checkpoints=${STABILITY_MAX_CHECKPOINTS:-1}
attempt_name=stability_mb${stability_micro_batch}_job${SLURM_JOB_ID}
invocation_id=${SLURM_JOB_ID}_${attempt_name}_steps1to3

BACE_EXP_ROOT=${STABILITY_ROOT} BACE_RUN_NAME=${attempt_name} \
  BACE_INVOCATION_ID=${invocation_id} ACTOR_MICRO_BATCH=${stability_micro_batch} \
  LOGPROB_MICRO_BATCH=${stability_micro_batch} SAVE_FREQ=-1 TEST_FREQ=-1 \
  BACE_RESUME_MODE=disable SEGMENT_END_STEP=3 MAX_CHECKPOINTS=${formal_max_checkpoints} \
  bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh"

metadata_dir=${STABILITY_ROOT}/run_metadata/${attempt_name}/${invocation_id}
artifact_dir=${STABILITY_ROOT}/bace_artifacts/${attempt_name}
log_file=${STABILITY_ROOT}/logs/${attempt_name}/${invocation_id}.log
report_dir=${STABILITY_ROOT}/profile_reports
mkdir -p "${report_dir}"

grep -Eq "training/global_step['\"]?:[[:space:]]*3|training/global_step[^0-9]+3" "${log_file}"
python -m recipe.bace_gigpo.validate_trace "${artifact_dir}" \
    --output "${report_dir}/${attempt_name}_trace_validation.json"
python "${SCRIPT_DIR}/validate_selected_worker_metrics.py" "${artifact_dir}" \
    --output "${report_dir}/${attempt_name}_selected_worker.json"
python "${SCRIPT_DIR}/analyze_host_resources.py" \
    "${metadata_dir}/host_resource_samples.csv" --max-pids 12000 \
    --warn-only \
    --output "${report_dir}/${attempt_name}_host_resources.json"
python "${SCRIPT_DIR}/check_formal_storage.py" "${FORMAL_ROOT}" \
    --expected-prefix /hpcwork/rwth2089/xsz96350 \
    --minimum-free-gib 64 \
    --output "${report_dir}/${attempt_name}_formal_storage.json"

# A repair invocation may name a segment cancelled before its first checkpoint.
# Preserve those partial traces, while ordinary fresh profiles skip this step.
if [[ -n "${ABORTED_FORMAL_JOB_ID:-}" ]]; then
    bash "${SCRIPT_DIR}/archive_aborted_formal_run.sh" \
        "${ABORTED_FORMAL_JOB_ID}"
fi

bash "${SCRIPT_DIR}/submit_alfworld_h100_4gpu_rolling.sh" \
    --target-step 150 \
    --micro-batch "${stability_micro_batch}" \
    --max-checkpoints "${formal_max_checkpoints}" \
    --dependency-job "${SLURM_JOB_ID}"
