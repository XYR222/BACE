#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
FORMAL_SCRIPT=${SCRIPT_DIR}/run_alfworld_h100_4gpu_rolling.sbatch
target_step=${TARGET_STEP:-150}
micro_batch=${ACTOR_MICRO_BATCH:-16}
max_checkpoints=${MAX_CHECKPOINTS:-1}
dependency_job=
run_name=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_main_seed0}
partition=${BACE_FORMAL_PARTITION:-c25g}
account=${BACE_FORMAL_ACCOUNT:-rwth2089}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-step) target_step=$2; shift 2 ;;
        --micro-batch) micro_batch=$2; shift 2 ;;
        --max-checkpoints) max_checkpoints=$2; shift 2 ;;
        --dependency-job) dependency_job=$2; shift 2 ;;
        --run-name) run_name=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --account) account=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ "${target_step}" =~ ^[0-9]+$ ]] && (( target_step >= 1 && target_step <= 150 ))
[[ "${micro_batch}" =~ ^(32|16|8)$ ]]
[[ "${max_checkpoints}" =~ ^(1|2)$ ]]
[[ "${run_name}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${partition}" =~ ^(c23g|c25g)$ ]]
[[ "${account}" =~ ^[A-Za-z0-9_-]+$ ]]
if [[ -n "${dependency_job}" ]]; then
    [[ "${dependency_job}" =~ ^[0-9]+$ ]]
fi

mkdir -p "${FORMAL_ROOT}/chain" "${FORMAL_ROOT}/slurm"
primary_args=(--parsable --partition="${partition}" --account="${account}")
if [[ -n "${dependency_job}" ]]; then
    primary_args+=(--dependency="afterok:${dependency_job}")
fi
export_spec="ALL,TARGET_STEP=${target_step},ACTOR_MICRO_BATCH=${micro_batch},MAX_CHECKPOINTS=${max_checkpoints},SAVE_FREQ=5,BACE_RUN_NAME=${run_name}"
primary_job_id=$(sbatch "${primary_args[@]}" --export="${export_spec}" "${FORMAL_SCRIPT}")
recovery_job_id=$(sbatch --parsable --partition="${partition}" --account="${account}" \
    --dependency="afternotok:${primary_job_id}" \
    --export="${export_spec}" "${FORMAL_SCRIPT}")

manifest=${FORMAL_ROOT}/chain/rolling_4gpu_${primary_job_id}.tsv
{
    printf 'role\tjob_id\tdependency\trun_name\tpartition\taccount\ttarget_step\tmicro_batch\tmax_checkpoints\tsave_freq\n'
    printf 'primary\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t5\n' \
        "${primary_job_id}" "${dependency_job:-none}" "${run_name}" \
        "${partition}" "${account}" "${target_step}" "${micro_batch}" "${max_checkpoints}"
    printf 'recovery\t%s\tafternotok:%s\t%s\t%s\t%s\t%s\t%s\t%s\t5\n' \
        "${recovery_job_id}" "${primary_job_id}" "${run_name}" \
        "${partition}" "${account}" "${target_step}" "${micro_batch}" "${max_checkpoints}"
} > "${manifest}"

printf 'primary_job_id=%s\nrecovery_job_id=%s\nmanifest=%s\n' \
    "${primary_job_id}" "${recovery_job_id}" "${manifest}"
