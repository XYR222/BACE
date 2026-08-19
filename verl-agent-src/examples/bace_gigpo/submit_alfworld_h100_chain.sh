#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
segment_steps=
micro_batch=
max_checkpoints=
dependency_job=
no_initial_dependency=0
source_profile_job=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --segment-steps) segment_steps=$2; shift 2 ;;
        --micro-batch) micro_batch=$2; shift 2 ;;
        --max-checkpoints) max_checkpoints=$2; shift 2 ;;
        --dependency-job) dependency_job=$2; shift 2 ;;
        --no-initial-dependency) no_initial_dependency=1; shift ;;
        --source-profile-job) source_profile_job=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${segment_steps}" =~ ^[0-9]+$ ]] && (( segment_steps >= 1 && segment_steps <= 20 ))
[[ "${micro_batch}" =~ ^(32|16|8)$ ]]
[[ "${max_checkpoints}" =~ ^(1|2)$ ]]
if (( no_initial_dependency )); then
    [[ -z "${dependency_job}" ]]
    [[ "${source_profile_job}" =~ ^[0-9]+$ ]]
else
    [[ "${dependency_job}" =~ ^[0-9]+$ ]]
    source_profile_job=${source_profile_job:-${dependency_job}}
fi

mkdir -p "${FORMAL_ROOT}/chain" "${FORMAL_ROOT}/slurm"
manifest=${FORMAL_ROOT}/chain/chain_from_profile_${source_profile_job}.tsv
printf 'segment_end_step\tjob_id\tdependency\tmicro_batch\tmax_checkpoints\n' > "${manifest}"

dependency=${dependency_job}
end_step=${segment_steps}
while (( end_step <= 150 )); do
    sbatch_args=(--parsable)
    dependency_label=none
    if [[ -n "${dependency}" ]]; then
        sbatch_args+=(--dependency="afterok:${dependency}")
        dependency_label=${dependency}
    fi
    job_id=$(sbatch "${sbatch_args[@]}" \
        --export="ALL,SEGMENT_END_STEP=${end_step},ACTOR_MICRO_BATCH=${micro_batch},MAX_CHECKPOINTS=${max_checkpoints}" \
        "${SCRIPT_DIR}/slurm_alfworld_h100_4gpu_segment.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${end_step}" "${job_id}" "${dependency_label}" "${micro_batch}" "${max_checkpoints}" \
        >> "${manifest}"
    dependency=${job_id}
    if (( end_step == 150 )); then break; fi
    end_step=$(( end_step + segment_steps ))
    (( end_step > 150 )) && end_step=150
done

echo "Formal chain submitted; manifest=${manifest}; final_job_id=${dependency}"
