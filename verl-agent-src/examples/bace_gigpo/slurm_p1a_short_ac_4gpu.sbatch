#!/usr/bin/env bash
#SBATCH --job-name=bace_p1a_ac4
#SBATCH --partition=c23g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
SOURCE_STEP=148
END_STEP=150
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${BACE_EXP_ROOT}/checkpoints/p1s_late_s3_stability_3252640_mb16/global_step_${SOURCE_STEP}}
JOB_TOKEN=${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}
GATE_ROOT=${BACE_EXP_ROOT}/p1a_short_ac/${JOB_TOKEN}
A_RUN="bace_exact_quota_current_rmin2_from148_diag_seed0_${JOB_TOKEN}"
C_RUN="bace_exact_quota_rmin4_from148_migration_diag_seed0_${JOB_TOKEN}"
mkdir -p "${GATE_ROOT}"

[[ -d "${SOURCE_CHECKPOINT}/actor" ]]
[[ -f "${SOURCE_CHECKPOINT}/data.pt" ]]
[[ -f "${SOURCE_CHECKPOINT}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/bace_collector_state.json")" == "${SOURCE_STEP}" ]]
for rank in 0 1 2 3; do
    for component in model optim extra_state; do
        shard="${SOURCE_CHECKPOINT}/actor/${component}_world_size_4_rank_${rank}.pt"
        [[ -s "${shard}" ]] || {
            echo "Incomplete source checkpoint: missing or empty ${shard}" >&2
            exit 2
        }
    done
done
find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | sort > "${GATE_ROOT}/source_checkpoint_before.tsv"

python "${SCRIPT_DIR}/check_formal_storage.py" "${BACE_EXP_ROOT}" \
    --expected-prefix /hpcwork/rwth2089/xsz96350 \
    --minimum-free-gib 32 \
    --output "${GATE_ROOT}/storage_preflight.json"

# Keep the complete formal GiGPO/BACE workload fixed between A and C.
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export ENV_MAX_STEPS=50
export MAX_RESPONSE_LENGTH=512
export PPO_MINI_BATCH=256
# The documented 32→16→8 OOM fallback is applied after the first profile
# attempt: the previous A attempt reached PPO backward with only ~9.4 GiB
# free, so both actor and log-prob micro batches are pinned to 16 here.
export ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-16}
export LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-16}
export ROLLOUT_TP=1
export GPU_MEMORY_UTILIZATION=0.60
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=128
export VAL_BEFORE_TRAIN=false
export LOGGER_BACKENDS="['console','tensorboard']"
export SEGMENT_END_STEP=${END_STEP}
export SAVE_FREQ=5
export TEST_FREQ=5
export BACE_RESUME_MODE=resume_path
# The manager observes the external step145 plus the new step150. Two slots
# prevent it from treating the read-only source as an old checkpoint to delete.
export MAX_CHECKPOINTS=2

cleanup_ray() {
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
}
trap cleanup_ray EXIT

run_arm() {
    local arm=$1
    local run_name=$2
    local min_roots=$3
    local migration_enabled=$4
    local migration_mode=$5

    export BACE_RUN_NAME=${run_name}
    export BACE_INVOCATION_ID="${JOB_TOKEN}_${arm}_from148_to150"
    export CHECKPOINT_DIR="${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}"
    export BACE_ARTIFACT_DIR="${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}"
    export ROLLOUT_DATA_DIR="${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}"
    export TENSORBOARD_DIR="${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}"
    export RUN_METADATA_DIR="${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}"
    export TRACE_VALIDATION_REPORT="${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.json"
    export LOG_FILE="${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log"

    [[ ! -e "${CHECKPOINT_DIR}" ]]
    bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
        algorithm.bace.min_natural_roots="${min_roots}" \
        algorithm.bace.branch_pool_mode=main_reuse \
        algorithm.bace.root_active_executor=true \
        algorithm.bace.checkpoint_migration.enabled="${migration_enabled}" \
        algorithm.bace.checkpoint_migration.mode="${migration_mode}"

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        cleanup_ray
        return
    fi

    [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]
    [[ "$(<"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")" == "${END_STEP}" ]]
    [[ "$(jq -r '.global_step' "${CHECKPOINT_DIR}/global_step_${END_STEP}/bace_collector_state.json")" == "${END_STEP}" ]]
    python "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep 1 --output "${GATE_ROOT}/${arm}_checkpoint_prune.json"
    python -m recipe.bace_gigpo.validate_trace "${BACE_ARTIFACT_DIR}" \
        --output "${GATE_ROOT}/${arm}_trace_validation.json"
    python "${SCRIPT_DIR}/validate_p1s_profile.py" \
        "${BACE_ARTIFACT_DIR}" --pool-mode main_reuse --root-active true \
        --output "${GATE_ROOT}/${arm}_p1s_validation.json"
    cleanup_ray
}

run_arm A "${A_RUN}" 2 false strict
run_arm C "${C_RUN}" 4 true diagnostic_rmin2_to4

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    cleanup_ray
    trap - EXIT
    exit 0
fi

[[ ! -e "${BACE_EXP_ROOT}/checkpoints/${A_RUN}/checkpoint_migration.json" ]]
C_MIGRATION="${BACE_EXP_ROOT}/checkpoints/${C_RUN}/checkpoint_migration.json"
python "${SCRIPT_DIR}/validate_p1a_short_ac.py" \
    --a "${BACE_EXP_ROOT}/bace_artifacts/${A_RUN}" \
    --c "${BACE_EXP_ROOT}/bace_artifacts/${C_RUN}" \
    --migration "${C_MIGRATION}" \
    --source-step "${SOURCE_STEP}" \
    --output "${GATE_ROOT}/ac_comparison.json"

find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | sort > "${GATE_ROOT}/source_checkpoint_after.tsv"
cmp "${GATE_ROOT}/source_checkpoint_before.tsv" "${GATE_ROOT}/source_checkpoint_after.tsv"

jq -n \
    --arg source "${SOURCE_CHECKPOINT}" \
    --arg a_run "${A_RUN}" \
    --arg c_run "${C_RUN}" \
    --arg comparison "${GATE_ROOT}/ac_comparison.json" \
    '{ok:true,scope:"mechanism_and_profile_only",source_checkpoint:$source,
      a_run:$a_run,c_run:$c_run,comparison:$comparison}' \
    > "${GATE_ROOT}/final_gate.json"

cleanup_ray
trap - EXIT
