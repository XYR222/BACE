#!/usr/bin/env bash
#SBATCH --job-name=bace15b_2g_s3_gate
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
FORMAL_SCRIPT=${SCRIPT_DIR}/run_alfworld_h100_2gpu_reference_aligned.sbatch
SOURCE_STEP=140
TARGET_STEP=141
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${EXP_ROOT}/preserved_checkpoints/bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry/global_step_${SOURCE_STEP}}
JOB_TOKEN=${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}
GATE_RUN_NAME=gate_2gpu_s3_from140_${JOB_TOKEN}
GATE_ROOT=${EXP_ROOT}/two_gpu_s3_gate/${JOB_TOKEN}
GATE_CHECKPOINT=${EXP_ROOT}/checkpoints/${GATE_RUN_NAME}/global_step_${TARGET_STEP}
GATE_MILESTONE=${EXP_ROOT}/preserved_checkpoints/${GATE_RUN_NAME}/global_step_${TARGET_STEP}
GATE_ARTIFACT=${EXP_ROOT}/bace_artifacts/${GATE_RUN_NAME}
GATE_TENSORBOARD=${EXP_ROOT}/tensorboard/${GATE_RUN_NAME}
FORMAL_RUN_NAME=${FORMAL_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_2gpu_s3_seed0_20260830}

mkdir -p "${GATE_ROOT}" "${EXP_ROOT}/slurm"
[[ -x "${FORMAL_SCRIPT}" || -f "${FORMAL_SCRIPT}" ]]
[[ ! -e "${EXP_ROOT}/checkpoints/${GATE_RUN_NAME}" ]]
[[ -s "${SOURCE_CHECKPOINT}/data.pt" ]]
[[ -s "${SOURCE_CHECKPOINT}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/bace_collector_state.json")" == "${SOURCE_STEP}" ]]
for rank in 0 1; do
    for component in model optim extra_state; do
        [[ -s "${SOURCE_CHECKPOINT}/actor/${component}_world_size_2_rank_${rank}.pt" ]]
    done
done

find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_before.tsv"

# The gate executes the same self-contained entry point and full GiGPO-aligned
# workload as the formal run.  Overrides below only isolate output, resume one
# late step, and avoid spending the one-hour allocation on validation rollout.
export BACE_SCHEDULING_PROFILE=optimized
export BACE_RUN_NAME=${GATE_RUN_NAME}
export BACE_INVOCATION_ID=${JOB_TOKEN}_from140_to141
export TARGET_STEP
export SAVE_FREQ=1
export MAX_CHECKPOINTS=1
export MILESTONE_CHECKPOINT_STEPS=${TARGET_STEP}

bash "${FORMAL_SCRIPT}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.validate_on_last_step=False

tracker=${EXP_ROOT}/checkpoints/${GATE_RUN_NAME}/latest_checkpointed_iteration.txt
[[ -s "${tracker}" ]]
[[ "$(<"${tracker}")" == "${TARGET_STEP}" ]]
[[ -s "${GATE_CHECKPOINT}/data.pt" ]]
[[ -s "${GATE_CHECKPOINT}/bace_collector_state.json" ]]
[[ "$(jq -r '.global_step' "${GATE_CHECKPOINT}/bace_collector_state.json")" == "${TARGET_STEP}" ]]
for rank in 0 1; do
    for component in model optim extra_state; do
        [[ -s "${GATE_CHECKPOINT}/actor/${component}_world_size_2_rank_${rank}.pt" ]]
    done
done

# The production run preserves 10,75,100,140.  The one-step gate uses its target
# step 141 to exercise exactly the same preservation path without pretending
# that the external source checkpoint was created by this invocation.
[[ -f "${GATE_MILESTONE}/.complete" ]]
[[ -s "${GATE_MILESTONE}/preservation_manifest.json" ]]
[[ "$(jq -r '.method' "${GATE_MILESTONE}/preservation_manifest.json")" == hardlink ]]
[[ "${GATE_CHECKPOINT}/data.pt" -ef "${GATE_MILESTONE}/data.pt" ]]
[[ "${GATE_CHECKPOINT}/bace_collector_state.json" -ef "${GATE_MILESTONE}/bace_collector_state.json" ]]
for rank in 0 1; do
    for component in model optim extra_state; do
        active_shard=${GATE_CHECKPOINT}/actor/${component}_world_size_2_rank_${rank}.pt
        milestone_shard=${GATE_MILESTONE}/actor/${component}_world_size_2_rank_${rank}.pt
        [[ -s "${milestone_shard}" ]]
        [[ "${active_shard}" -ef "${milestone_shard}" ]]
    done
done

python3 -m recipe.bace_gigpo.validate_trace "${GATE_ARTIFACT}" \
    --output "${GATE_ROOT}/trace_validation.json"
python3 "${SCRIPT_DIR}/validate_p1s_profile.py" "${GATE_ARTIFACT}" \
    --pool-mode main_reuse --root-active true \
    --output "${GATE_ROOT}/p1s_validation.json"
python3 "${SCRIPT_DIR}/validate_selected_worker_metrics.py" "${GATE_ARTIFACT}" \
    --output "${GATE_ROOT}/selected_worker_validation.json"

# Pairwise-Fixed/Stopping are separate acquisition experiments and were not
# part of the original 2-GPU BACE run.  The compatibility default named
# ``full`` follows the historical one-round Exact allocation; require that no
# pairwise execution flag was active in the actual gate trace.
latest_summary=$(find "${GATE_ARTIFACT}" -name summary.json -type f | LC_ALL=C sort | tail -1)
[[ -s "${latest_summary}" ]]
jq -e '
    ((.bace_metrics.pairwise_fixed // 0) == 0) and
    ((.bace_metrics.pairwise_stopping // 0) == 0)
' "${latest_summary}" >/dev/null

find "${GATE_TENSORBOARD}" -type f -name 'events.out.tfevents.*' -size +0c \
    -print -quit | grep -q .
find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${GATE_ROOT}/source_after.tsv"
cmp "${GATE_ROOT}/source_before.tsv" "${GATE_ROOT}/source_after.tsv"

# A fresh formal name is mandatory: automatic resume is useful only after this
# job has been created, never as an accidental continuation of an older run.
formal_checkpoint_root=${EXP_ROOT}/checkpoints/${FORMAL_RUN_NAME}
formal_milestone_root=${EXP_ROOT}/preserved_checkpoints/${FORMAL_RUN_NAME}
if [[ -e "${formal_checkpoint_root}" || -e "${formal_milestone_root}" ]]; then
    printf 'Refusing to submit: formal checkpoint or milestone path already exists: %s / %s\n' \
        "${formal_checkpoint_root}" "${formal_milestone_root}" >&2
    exit 3
fi

# Export only formal-run selectors.  In particular, do not leak the gate's
# MILESTONE_CHECKPOINT_STEPS=141 or its invocation ID into production; the
# formal entry point then applies its audited 10,75,100,140 milestone default.
formal_job_id=$(sbatch --parsable \
    --export=BACE_SCHEDULING_PROFILE=optimized,BACE_RUN_NAME="${FORMAL_RUN_NAME}",TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2 \
    "${FORMAL_SCRIPT}")
printf '%s\n' "${formal_job_id}" > "${GATE_ROOT}/formal_job_id.txt"

jq -n \
    --arg gate_job_id "${JOB_TOKEN}" \
    --arg formal_job_id "${formal_job_id}" \
    --arg source_checkpoint "${SOURCE_CHECKPOINT}" \
    --arg target_checkpoint "${GATE_CHECKPOINT}" \
    --arg tensorboard_dir "${GATE_TENSORBOARD}" \
    --arg formal_run_name "${FORMAL_RUN_NAME}" \
    '{ok:true, gate_job_id:$gate_job_id, formal_job_id:$formal_job_id,
      source_checkpoint:$source_checkpoint, source_unchanged:true,
      target_checkpoint:$target_checkpoint, tensorboard_dir:$tensorboard_dir,
      formal_run_name:$formal_run_name, scheduling_profile:"optimized",
      branch_pool_mode:"main_reuse", root_active_executor:true,
      local_credit_mode:"occurrence", pairwise_enabled:false,
      formal_milestone_steps:[10,75,100,140], gate_milestone_step:141}' \
    > "${GATE_ROOT}/final_gate.json"

printf 'Gate passed; submitted formal 2-GPU S3 job %s (%s).\n' \
    "${formal_job_id}" "${FORMAL_RUN_NAME}"
