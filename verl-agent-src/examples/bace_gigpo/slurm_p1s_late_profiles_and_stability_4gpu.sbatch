#!/usr/bin/env bash
#SBATCH --job-name=bace_p1s_late4
#SBATCH --partition=c23g
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

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
SOURCE_STEP=145
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${BACE_EXP_ROOT}/checkpoints/bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0/global_step_${SOURCE_STEP}}
PROFILE_END_STEP=146
STABILITY_END_STEP=148
JOB_TOKEN=${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}
GATE_ROOT=${BACE_EXP_ROOT}/p1s_late_gate/${JOB_TOKEN}
mkdir -p "${GATE_ROOT}"

[[ -d "${SOURCE_CHECKPOINT}/actor" ]]
[[ -f "${SOURCE_CHECKPOINT}/data.pt" ]]
[[ -f "${SOURCE_CHECKPOINT}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/bace_collector_state.json")" == "${SOURCE_STEP}" ]]
for rank in 0 1 2 3; do
    [[ -f "${SOURCE_CHECKPOINT}/actor/model_world_size_4_rank_${rank}.pt" ]]
    [[ -f "${SOURCE_CHECKPOINT}/actor/optim_world_size_4_rank_${rank}.pt" ]]
    [[ -f "${SOURCE_CHECKPOINT}/actor/extra_state_world_size_4_rank_${rank}.pt" ]]
done
find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_before.tsv"

python "${SCRIPT_DIR}/check_formal_storage.py" "${BACE_EXP_ROOT}" \
    --expected-prefix /hpcwork/rwth2089/xsz96350 \
    --minimum-free-gib 32 \
    --output "${GATE_ROOT}/storage_preflight.json"

# Formal per-step workload. Profiles disable validation/checkpoint writes only;
# model, data, root budget, horizon, response length, and PPO parameters remain
# identical to the 4-GPU production configuration.
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export ENV_MAX_STEPS=50
export MAX_RESPONSE_LENGTH=512
export PPO_MINI_BATCH=256
# Job 3237296 established that the formal late-step workload OOMs during the
# actor backward pass at 32.  Keep all four A/B profiles on one micro-batch;
# if 16 also has a confirmed CUDA OOM, rerun the complete set at 8.
INITIAL_MICRO_BATCH=${INITIAL_MICRO_BATCH:-16}
case "${INITIAL_MICRO_BATCH}" in
    16|8) ;;
    *) echo "INITIAL_MICRO_BATCH must be 16 or 8" >&2; exit 2 ;;
esac
export ROLLOUT_TP=1
export GPU_MEMORY_UTILIZATION=0.60
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=128
export VAL_BEFORE_TRAIN=false
export LOGGER_BACKENDS="['console']"
export MAX_CHECKPOINTS=1
export TEST_FREQ=-1
export BACE_RESUME_MODE=resume_path

cleanup_ray() {
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
}
trap cleanup_ray EXIT

run_profile() {
    local label=$1
    local pool_mode=$2
    local root_active=$3
    local micro_batch=$4
    export ACTOR_MICRO_BATCH=${micro_batch}
    export LOGPROB_MICRO_BATCH=${micro_batch}
    export BACE_RUN_NAME="p1s_late_${label}_${JOB_TOKEN}_mb${micro_batch}"
    export BACE_INVOCATION_ID="${JOB_TOKEN}_${label}_mb${micro_batch}"
    export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}"
    export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}"
    export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}"
    export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}"
    export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}"
    export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.json"
    export LOG_FILE="${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log"
    export SEGMENT_END_STEP=${PROFILE_END_STEP}
    export SAVE_FREQ=-1

    set +e
    bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
        algorithm.bace.branch_pool_mode="${pool_mode}" \
        algorithm.bace.root_active_executor="${root_active}"
    local train_status=$?
    set -e

    if (( train_status != 0 )); then
        cleanup_ray
        if [[ -f "${LOG_FILE}" ]] && grep -Eiq \
            'torch\.OutOfMemoryError|CUDA out of memory|RayTaskError\(OutOfMemoryError\)' \
            "${LOG_FILE}"; then
            echo "Confirmed CUDA OOM: profile=${label} micro_batch=${micro_batch}" >&2
            return 42
        fi
        echo "Profile ${label} failed with non-OOM status ${train_status}; not retrying" >&2
        return "${train_status}"
    fi

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        cleanup_ray
        return
    fi

    python -m recipe.bace_gigpo.validate_trace \
        "${BACE_ARTIFACT_DIR}" \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_final.json"
    python "${SCRIPT_DIR}/validate_p1s_profile.py" \
        "${BACE_ARTIFACT_DIR}" --pool-mode "${pool_mode}" \
        --root-active "${root_active}" \
        --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_p1s.json"
    cleanup_ray
}

declare -A PROFILE_ARTIFACTS
run_profile_set() {
    local micro_batch=$1
    local label pool_mode root_active status
    while read -r label pool_mode root_active; do
        if run_profile "${label}" "${pool_mode}" "${root_active}" "${micro_batch}"; then
            PROFILE_ARTIFACTS[${label}]="${BACE_EXP_ROOT}/bace_artifacts/p1s_late_${label}_${JOB_TOKEN}_mb${micro_batch}"
        else
            status=$?
            return "${status}"
        fi
    done <<'EOF'
s0 dedicated false
s1 main_reuse false
s2 dedicated true
s3 main_reuse true
EOF
}

profile_micro_batch=${INITIAL_MICRO_BATCH}
while true; do
    echo "Running complete S0-S3 profile set with micro-batch ${profile_micro_batch}"
    if run_profile_set "${profile_micro_batch}"; then
        break
    else
        status=$?
    fi
    if (( status == 42 && profile_micro_batch == 16 )); then
        profile_micro_batch=8
        echo "Restarting the complete S0-S3 set at micro-batch 8 for a fair comparison"
        continue
    fi
    exit "${status}"
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
        -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_after.tsv"
    cmp "${GATE_ROOT}/source_checkpoint_before.tsv" "${GATE_ROOT}/source_checkpoint_after.tsv"
    cleanup_ray
    trap - EXIT
    exit 0
fi

python "${SCRIPT_DIR}/validate_p1s_late_profiles.py" \
    --s0 "${PROFILE_ARTIFACTS[s0]}" \
    --s1 "${PROFILE_ARTIFACTS[s1]}" \
    --s2 "${PROFILE_ARTIFACTS[s2]}" \
    --s3 "${PROFILE_ARTIFACTS[s3]}" \
    --expected-step "${PROFILE_END_STEP}" \
    --output "${GATE_ROOT}/profile_comparison.json" \
    --selection-output "${GATE_ROOT}/selected_mode.txt"

selected_mode=$(<"${GATE_ROOT}/selected_mode.txt")
case "${selected_mode}" in
    s1) stability_pool=main_reuse; stability_root_active=false ;;
    s3) stability_pool=main_reuse; stability_root_active=true ;;
    *) echo "Unexpected selected mode: ${selected_mode}" >&2; exit 2 ;;
esac

run_stability() {
    local micro_batch=$1
    export ACTOR_MICRO_BATCH=${micro_batch}
    export LOGPROB_MICRO_BATCH=${micro_batch}
    export BACE_RUN_NAME="p1s_late_${selected_mode}_stability_${JOB_TOKEN}_mb${micro_batch}"
    export BACE_INVOCATION_ID="${JOB_TOKEN}_${selected_mode}_stability_mb${micro_batch}"
    export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}"
    export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}"
    export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}"
    export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}"
    export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}"
    export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.json"
    export LOG_FILE="${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log"
    export SEGMENT_END_STEP=${STABILITY_END_STEP}
    export SAVE_FREQ=3
    # The FSDP manager seeds retention bookkeeping from the external step-145
    # actor. Two slots protect that read-only source plus the new stability actor.
    # We prune only the isolated output root below.
    export MAX_CHECKPOINTS=2

    set +e
    bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
        algorithm.bace.branch_pool_mode="${stability_pool}" \
        algorithm.bace.root_active_executor="${stability_root_active}"
    local train_status=$?
    set -e
    if (( train_status != 0 )); then
        cleanup_ray
        if [[ -f "${LOG_FILE}" ]] && grep -Eiq \
            'torch\.OutOfMemoryError|CUDA out of memory|RayTaskError\(OutOfMemoryError\)' \
            "${LOG_FILE}"; then
            echo "Confirmed CUDA OOM: stability micro_batch=${micro_batch}" >&2
            return 42
        fi
        echo "Stability run failed with non-OOM status ${train_status}; not retrying" >&2
        return "${train_status}"
    fi
}

stability_micro_batch=${profile_micro_batch}
while true; do
    if run_stability "${stability_micro_batch}"; then
        break
    else
        status=$?
    fi
    if (( status == 42 && stability_micro_batch == 16 )); then
        stability_micro_batch=8
        echo "Retrying the complete three-step stability run at micro-batch 8"
        continue
    fi
    exit "${status}"
done

[[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]
[[ "$(<"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")" == "${STABILITY_END_STEP}" ]]
[[ "$(jq -r '.global_step' "${CHECKPOINT_DIR}/global_step_${STABILITY_END_STEP}/bace_collector_state.json")" == "${STABILITY_END_STEP}" ]]
python "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
    --keep 1 --output "${GATE_ROOT}/stability_checkpoint_prune.json"
summary_count=$(find "${BACE_ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name summary.json -print | wc -l)
[[ "${summary_count//[[:space:]]/}" == 3 ]]
if find "${BACE_ARTIFACT_DIR}" -mindepth 2 -maxdepth 2 -name summary.json \
    -exec jq -e '.status == "complete"' {} \; | grep -q false; then
    echo "A stability summary is incomplete" >&2
    exit 1
fi
python -m recipe.bace_gigpo.validate_trace \
    "${BACE_ARTIFACT_DIR}" \
    --output "${GATE_ROOT}/stability_trace_validation.json"

find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_after.tsv"
cmp "${GATE_ROOT}/source_checkpoint_before.tsv" "${GATE_ROOT}/source_checkpoint_after.tsv"

jq -n \
    --arg selected_mode "${selected_mode}" \
    --argjson profile_micro_batch "${profile_micro_batch}" \
    --argjson stability_micro_batch "${stability_micro_batch}" \
    --arg source_checkpoint "${SOURCE_CHECKPOINT}" \
    --arg stability_checkpoint "${CHECKPOINT_DIR}/global_step_${STABILITY_END_STEP}" \
    --arg profile_report "${GATE_ROOT}/profile_comparison.json" \
    '{ok:true, selected_mode:$selected_mode,
      profile_micro_batch:$profile_micro_batch, stability_micro_batch:$stability_micro_batch,
      source_checkpoint:$source_checkpoint,
      stability_checkpoint:$stability_checkpoint, profile_report:$profile_report}' \
    > "${GATE_ROOT}/final_gate.json"

cleanup_ray
trap - EXIT
