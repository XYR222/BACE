#!/usr/bin/env bash
# Archive selected completed checkpoints while an independent trainer job runs.
# This job uses CPU only and never touches the active checkpoint directories.
# Usage: sbatch --export=ALL,BACE_RUN_NAME=... this-file
#
# The source checkpoint is considered safe only after the trainer tracker has
# advanced past it and all actor/data/BACE state files exist.  A temporary
# destination and atomic rename prevent readers from seeing a partial copy.
#SBATCH --job-name=bace_ckpt_archive
# The archive watcher is CPU-only.  Use the one-hour devel partition and chain
# continuation allocations; this avoids consuming a GPU merely to poll files.
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

BASE=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
RUN_NAME=${BACE_RUN_NAME:?set BACE_RUN_NAME to the active training run}
TRAIN_JOB_ID=${BACE_TRAIN_JOB_ID:?set BACE_TRAIN_JOB_ID to the active training job}
TARGET_STEPS=${BACE_TARGET_STEPS:-25 75 100 140}
SOURCE_ROOT=${BASE}/checkpoints/${RUN_NAME}
ARCHIVE_ROOT=${BASE}/preserved_checkpoints/${RUN_NAME}
POLL_SECONDS=${BACE_ARCHIVE_POLL_SECONDS:-30}
WATCH_SECONDS=${BACE_ARCHIVE_WATCH_SECONDS:-3300}
WATCH_DEADLINE=$(( $(date +%s) + WATCH_SECONDS ))

mkdir -p "${ARCHIVE_ROOT}"
printf 'run=%s\ntrain_job=%s\ntarget_steps=%s\n' \
    "${RUN_NAME}" "${TRAIN_JOB_ID}" "${TARGET_STEPS}" \
    > "${ARCHIVE_ROOT}/archive_job_${SLURM_JOB_ID}.metadata"

checkpoint_complete() {
    local path=$1
    [[ -d "${path}/actor" ]] || return 1
    [[ -f "${path}/data.pt" ]] || return 1
    [[ -f "${path}/bace_collector_state.json" ]] || return 1
    local required
    for required in \
        model_world_size_2_rank_0.pt model_world_size_2_rank_1.pt \
        optim_world_size_2_rank_0.pt optim_world_size_2_rank_1.pt \
        extra_state_world_size_2_rank_0.pt extra_state_world_size_2_rank_1.pt; do
        [[ -s "${path}/actor/${required}" ]] || return 1
    done
}

tracker=${SOURCE_ROOT}/latest_checkpointed_iteration.txt
while :; do
    latest=0
    if [[ -s "${tracker}" ]] && [[ $(<"${tracker}") =~ ^[0-9]+$ ]]; then
        latest=$(<"${tracker}")
    fi

    pending=0
    for step in ${TARGET_STEPS}; do
        dst=${ARCHIVE_ROOT}/global_step_${step}
        [[ -f "${dst}/.complete" ]] && continue
        pending=1
        src=${SOURCE_ROOT}/global_step_${step}
        if (( latest < step )) || ! checkpoint_complete "${src}"; then
            continue
        fi

        tmp=${ARCHIVE_ROOT}/.global_step_${step}.tmp-${SLURM_JOB_ID}-$$
        rm -rf "${tmp}"
        mkdir -p "${tmp}"
        rsync -a --numeric-ids "${src}/" "${tmp}/"
        checkpoint_complete "${tmp}" || {
            rm -rf "${tmp}"
            printf 'incomplete copy for step %s; retrying\n' "${step}" >&2
            continue
        }
        touch "${tmp}/.complete"
        mv "${tmp}" "${dst}"
        printf 'archived step %s to %s\n' "${step}" "${dst}"
    done

    (( pending == 0 )) && exit 0
    if ! squeue -h -j "${TRAIN_JOB_ID}" | grep -q .; then
        # The trainer has left the queue.  A final pass above is sufficient;
        # report missing requested targets explicitly rather than looping.
        missing=()
        for step in ${TARGET_STEPS}; do
            [[ -f "${ARCHIVE_ROOT}/global_step_${step}/.complete" ]] || missing+=("${step}")
        done
        if (( ${#missing[@]} )); then
            printf 'trainer ended before targets were archived: %s\n' "${missing[*]}" >&2
            exit 1
        fi
        exit 0
    fi
    now=$(date +%s)
    if (( now >= WATCH_DEADLINE )); then
        # devel allocations are limited to one hour.  Chain a continuation so
        # monitoring survives the allocation boundary without a user session.
        next_job=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
            --export="ALL,BACE_RUN_NAME=${RUN_NAME},BACE_TRAIN_JOB_ID=${TRAIN_JOB_ID}" "$0")
        printf 'watch_window_expired; continuation_job=%s\n' "${next_job}"
        exit 0
    fi
    remaining=$(( WATCH_DEADLINE - now ))
    (( remaining < POLL_SECONDS )) && sleep_seconds=${remaining} || sleep_seconds=${POLL_SECONDS}
    sleep "${sleep_seconds}"
done
