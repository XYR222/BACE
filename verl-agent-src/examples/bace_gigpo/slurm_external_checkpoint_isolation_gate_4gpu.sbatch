#!/usr/bin/env bash
#SBATCH --job-name=bace_ckpt_iso4
#SBATCH --partition=c23g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
SOURCE_STEP=148
END_STEP=149
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${BACE_EXP_ROOT}/checkpoints/p1s_late_s3_stability_3252640_mb16/global_step_${SOURCE_STEP}}
JOB_TOKEN=${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_NAME="checkpoint_isolation_gate_from148_${JOB_TOKEN}"
GATE_ROOT=${BACE_EXP_ROOT}/checkpoint_isolation_gate/${JOB_TOKEN}

export BACE_RUN_NAME=${RUN_NAME}
export BACE_INVOCATION_ID="${JOB_TOKEN}_from148_to149"
export CHECKPOINT_DIR=${BACE_EXP_ROOT}/checkpoints/${RUN_NAME}
export BACE_ARTIFACT_DIR=${BACE_EXP_ROOT}/bace_artifacts/${RUN_NAME}
export ROLLOUT_DATA_DIR=${BACE_EXP_ROOT}/rollout_trajectories/${RUN_NAME}
export TENSORBOARD_DIR=${BACE_EXP_ROOT}/tensorboard/${RUN_NAME}
export RUN_METADATA_DIR=${BACE_EXP_ROOT}/run_metadata/${RUN_NAME}/${BACE_INVOCATION_ID}
export TRACE_VALIDATION_REPORT=${BACE_EXP_ROOT}/trace_validation/${RUN_NAME}/${BACE_INVOCATION_ID}.json
export LOG_FILE=${BACE_EXP_ROOT}/logs/${RUN_NAME}/${BACE_INVOCATION_ID}.log

mkdir -p "${GATE_ROOT}"
[[ ! -e "${CHECKPOINT_DIR}" ]]
[[ -d "${SOURCE_CHECKPOINT}/actor" ]]
[[ -s "${SOURCE_CHECKPOINT}/data.pt" ]]
[[ -s "${SOURCE_CHECKPOINT}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/bace_collector_state.json")" == "${SOURCE_STEP}" ]]
for rank in 0 1 2 3; do
    for component in model optim extra_state; do
        shard=${SOURCE_CHECKPOINT}/actor/${component}_world_size_4_rank_${rank}.pt
        [[ -s "${shard}" ]] || {
            echo "Incomplete source checkpoint: ${shard}" >&2
            exit 2
        }
    done
done

find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_before.tsv"

python "${SCRIPT_DIR}/check_formal_storage.py" "${BACE_EXP_ROOT}" \
    --expected-prefix /hpcwork/rwth2089/xsz96350 \
    --minimum-free-gib 32 \
    --output "${GATE_ROOT}/storage_preflight.json"

# Keep the production per-step workload and the proven safe micro-batch.  This
# gate changes only the output root and runs exactly one resumed training step.
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export ENV_MAX_STEPS=50
export MAX_RESPONSE_LENGTH=512
export PPO_MINI_BATCH=256
export ACTOR_MICRO_BATCH=16
export LOGPROB_MICRO_BATCH=16
export ROLLOUT_TP=1
export GPU_MEMORY_UTILIZATION=0.60
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=128
export VAL_BEFORE_TRAIN=false
export LOGGER_BACKENDS="['console']"
export SEGMENT_END_STEP=${END_STEP}
export SAVE_FREQ=1
export TEST_FREQ=-1
export BACE_RESUME_MODE=resume_path
# One slot deliberately reproduces the destructive case from job 3252640.
# The fixed manager must discard the cross-root source from rotation candidates.
export MAX_CHECKPOINTS=1

cleanup_ray() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
    algorithm.bace.branch_pool_mode=main_reuse \
    algorithm.bace.root_active_executor=true

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
        -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_after.tsv"
    cmp "${GATE_ROOT}/source_checkpoint_before.tsv" "${GATE_ROOT}/source_checkpoint_after.tsv"
    cleanup_ray
    trap - EXIT
    exit 0
fi

TARGET=${CHECKPOINT_DIR}/global_step_${END_STEP}
[[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]
[[ "$(<"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")" == "${END_STEP}" ]]
[[ -s "${TARGET}/data.pt" ]]
[[ -s "${TARGET}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${TARGET}/bace_collector_state.json")" == "${END_STEP}" ]]
for rank in 0 1 2 3; do
    for component in model optim extra_state; do
        [[ -s "${TARGET}/actor/${component}_world_size_4_rank_${rank}.pt" ]]
    done
done

python -m recipe.bace_gigpo.validate_trace "${BACE_ARTIFACT_DIR}" \
    --output "${GATE_ROOT}/trace_validation.json"
python "${SCRIPT_DIR}/validate_p1s_profile.py" \
    "${BACE_ARTIFACT_DIR}" --pool-mode main_reuse --root-active true \
    --output "${GATE_ROOT}/p1s_validation.json"

find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_checkpoint_after.tsv"
cmp "${GATE_ROOT}/source_checkpoint_before.tsv" "${GATE_ROOT}/source_checkpoint_after.tsv"

jq -n \
    --arg source_checkpoint "${SOURCE_CHECKPOINT}" \
    --arg target_checkpoint "${TARGET}" \
    --argjson source_step "${SOURCE_STEP}" \
    --argjson target_step "${END_STEP}" \
    '{ok:true, source_unchanged:true, source_step:$source_step,
      target_step:$target_step, source_checkpoint:$source_checkpoint,
      target_checkpoint:$target_checkpoint, max_checkpoints:1}' \
    > "${GATE_ROOT}/final_gate.json"

cleanup_ray
trap - EXIT
