#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
RUN_NAME=bace_alfworld_qwen2_5_1_5b_exact_main_seed0
ABORTED_JOB_ID=${1:?aborted Slurm job ID is required}
checkpoint_root=${FORMAL_ROOT}/checkpoints/${RUN_NAME}

if [[ -f "${checkpoint_root}/latest_checkpointed_iteration.txt" ]]; then
    echo "Refusing to archive a formal run with a resumable checkpoint tracker." >&2
    exit 1
fi

archive_root=${FORMAL_ROOT}/aborted_runs/job_${ABORTED_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${archive_root}"
moved=0
for category in checkpoints bace_artifacts rollout_trajectories tensorboard; do
    source_path=${FORMAL_ROOT}/${category}/${RUN_NAME}
    if [[ -e "${source_path}" ]]; then
        mkdir -p "${archive_root}/${category}"
        mv "${source_path}" "${archive_root}/${category}/${RUN_NAME}"
        moved=1
    fi
done

printf 'aborted_job_id=%s\narchived_at=%s\nsource_run_name=%s\n' \
    "${ABORTED_JOB_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_NAME}" \
    > "${archive_root}/archive_manifest.txt"
echo "Archived incomplete formal state to ${archive_root} (moved=${moved})."
