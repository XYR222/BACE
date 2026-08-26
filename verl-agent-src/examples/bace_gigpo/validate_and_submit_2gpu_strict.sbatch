#!/usr/bin/env bash
#SBATCH --job-name=bace2g_strict_check
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
STRICT_SCRIPT=${SCRIPT_DIR}/run_alfworld_h100_2gpu_strict.sbatch

: "${DEBUG_JOB_ID:?DEBUG_JOB_ID must identify the ordinary one-step job}"
: "${DEBUG_RUN_NAME:?DEBUG_RUN_NAME is required}"
: "${BRANCH_JOB_ID:?BRANCH_JOB_ID must identify the branch-specialized job}"
: "${BRANCH_RUN_NAME:?BRANCH_RUN_NAME is required}"

source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WORKSPACE_ROOT}/verl-agent"
cd "${REPO_ROOT}"

validate_common() {
    local job_id=$1
    local run_name=$2
    local invocation_id=${job_id}_2gpu_strict
    local artifact_dir=${EXP_ROOT}/bace_artifacts/${run_name}
    local metadata_dir=${EXP_ROOT}/run_metadata/${run_name}/${invocation_id}
    local log_file=${EXP_ROOT}/logs/${run_name}/${invocation_id}.log
    local report_dir=${EXP_ROOT}/trace_validation/${run_name}
    local state exit_code

    read -r state exit_code < <(
        sacct -X -j "${job_id}" -n -P -o State,ExitCode |
            awk -F'|' 'NF >= 2 {print $1, $2; exit}'
    )
    [[ "${state}" == COMPLETED ]] || {
        printf 'Job %s did not complete: state=%s exit=%s\n' \
            "${job_id}" "${state}" "${exit_code}" >&2
        return 1
    }
    [[ "${exit_code}" == 0:0 ]] || {
        printf 'Job %s has non-zero exit code: %s\n' \
            "${job_id}" "${exit_code}" >&2
        return 1
    }
    [[ -s "${log_file}" ]] || {
        printf 'Training log is missing or empty: %s\n' "${log_file}" >&2
        return 1
    }
    [[ -s "${metadata_dir}/resolved_command.sh" ]] || {
        printf 'Resolved command is missing: %s\n' \
            "${metadata_dir}/resolved_command.sh" >&2
        return 1
    }

    grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" \
        "${log_file}"
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError' \
        "${log_file}"; then
        printf 'OOM marker found in %s\n' "${log_file}" >&2
        return 1
    fi

    grep -Fq 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'actor_rollout_ref.rollout.tensor_model_parallel_size=2' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'trainer.n_gpus_per_node=2' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'algorithm.bace.variant=batch_erv_exact' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'algorithm.bace.staged_root_batching=packed' \
        "${metadata_dir}/resolved_command.sh"
    grep -Fq 'algorithm.bace.branch_execution_mode=selected_worker' \
        "${metadata_dir}/resolved_command.sh"

    mkdir -p "${report_dir}"
    python -m recipe.bace_gigpo.validate_trace "${artifact_dir}" \
        --output "${report_dir}/${invocation_id}_final.json"
}

validate_common "${DEBUG_JOB_ID}" "${DEBUG_RUN_NAME}"
validate_common "${BRANCH_JOB_ID}" "${BRANCH_RUN_NAME}"

branch_artifact_dir=${EXP_ROOT}/bace_artifacts/${BRANCH_RUN_NAME}
branch_report_dir=${EXP_ROOT}/trace_validation/${BRANCH_RUN_NAME}
python "${SCRIPT_DIR}/validate_selected_worker_metrics.py" \
    "${branch_artifact_dir}" \
    --output "${branch_report_dir}/${BRANCH_JOB_ID}_2gpu_strict_selected_worker.json"

# This is the only place where the formal two-GPU job is submitted. Reaching
# it proves that both GPU jobs and every trace/selected-worker assertion above
# succeeded under set -e. No training override is passed to the strict script.
formal_job_id=$(sbatch --parsable "${STRICT_SCRIPT}")
recovery_job_id=$(sbatch --parsable --dependency="afternotok:${formal_job_id}" \
    "${STRICT_SCRIPT}")

chain_dir=${EXP_ROOT}/chain
mkdir -p "${chain_dir}"
manifest=${chain_dir}/two_gpu_strict_gate_${SLURM_JOB_ID}.txt
{
    printf 'debug_job_id=%s\n' "${DEBUG_JOB_ID}"
    printf 'debug_run_name=%s\n' "${DEBUG_RUN_NAME}"
    printf 'branch_job_id=%s\n' "${BRANCH_JOB_ID}"
    printf 'branch_run_name=%s\n' "${BRANCH_RUN_NAME}"
    printf 'validation_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'validation_status=passed\n'
    printf 'formal_job_id=%s\n' "${formal_job_id}"
    printf 'recovery_job_id=%s\n' "${recovery_job_id}"
    printf 'recovery_dependency=afternotok:%s\n' "${formal_job_id}"
    printf 'checkpoint_save_freq=5\n'
    printf 'checkpoint_keep=1\n'
    printf 'target_step=150\n'
    printf 'formal_script=%s\n' "${STRICT_SCRIPT}"
    printf 'formal_script_sha256=%s\n' "$(sha256sum "${STRICT_SCRIPT}" | awk '{print $1}')"
    printf 'formal_submission=no_hydra_overrides\n'
} > "${manifest}"

printf 'All two-GPU strict gates passed; formal job_id=%s; recovery_job_id=%s; manifest=%s\n' \
    "${formal_job_id}" "${recovery_job_id}" "${manifest}"
