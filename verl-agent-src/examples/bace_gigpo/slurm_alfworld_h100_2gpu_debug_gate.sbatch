#!/usr/bin/env bash
#SBATCH --job-name=bace15b_2g_gate
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_RUN_NAME=selected_worker_2gpu_gigpo_gate_${SLURM_JOB_ID}
export BACE_INVOCATION_ID=${SLURM_JOB_ID}_2gpu_gigpo_gate
# Gate 3082011 proved that the original value 32 OOMs in update_actor. Retry
# with the first approved execution-chunk fallback while preserving the global
# batch, PPO mini-batch and all other GiGPO/BACE parameters.
export ACTOR_MICRO_BATCH=16
export LOGPROB_MICRO_BATCH=16
unset CHECKPOINT_DIR BACE_ARTIFACT_DIR ROLLOUT_DATA_DIR TENSORBOARD_DIR
unset RUN_METADATA_DIR TRACE_VALIDATION_REPORT LOG_FILE

# Keep the complete two-GPU GiGPO configuration, but stop after one optimizer
# step. The disposable prior override ensures that this step exercises branch
# replay and selected-worker compaction instead of testing roots only.
bash "${SCRIPT_DIR}/run_alfworld_h100_2gpu.sh" \
    trainer.total_training_steps=1 \
    trainer.test_freq=-1 \
    algorithm.bace.history_initial_mean=0.60

artifact_dir=${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}
metadata_dir=${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}
log_file=${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}.log
report_dir=${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}
resolved_command=${metadata_dir}/resolved_command.sh

grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" \
    "${log_file}"
grep -Fq 'trainer.n_gpus_per_node=2' "${resolved_command}"
grep -Fq 'actor_rollout_ref.rollout.tensor_model_parallel_size=2' "${resolved_command}"
grep -Fq 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16' "${resolved_command}"
grep -Fq 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16' "${resolved_command}"
grep -Fq 'trainer.val_before_train=true' "${resolved_command}"
grep -Fq 'algorithm.bace.staged_root_batching=packed' "${resolved_command}"
grep -Fq 'algorithm.bace.branch_execution_mode=selected_worker' "${resolved_command}"

python -m recipe.bace_gigpo.validate_trace "${artifact_dir}" \
    --output "${report_dir}/${BACE_INVOCATION_ID}_final.json"
python "${SCRIPT_DIR}/validate_selected_worker_metrics.py" "${artifact_dir}" \
    --output "${report_dir}/${BACE_INVOCATION_ID}_selected_worker.json"

# Submission occurs only after training, configuration, trace and selected-
# worker checks above have all succeeded under `set -e`.
formal_job_id=$(sbatch --parsable "${SCRIPT_DIR}/slurm_alfworld_h100_2gpu_gigpo.sbatch")
chain_dir=${BACE_EXP_ROOT}/chain
mkdir -p "${chain_dir}"
manifest=${chain_dir}/two_gpu_gigpo_from_gate_${SLURM_JOB_ID}.txt
{
    printf 'debug_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'debug_run_name=%s\n' "${BACE_RUN_NAME}"
    printf 'debug_status=passed\n'
    printf 'original_micro_batch=32\n'
    printf 'effective_micro_batch=16\n'
    printf 'fallback_reason=confirmed_cuda_oom_in_gate_3082011\n'
    printf 'formal_job_id=%s\n' "${formal_job_id}"
    printf 'formal_submission=after_all_debug_gates\n'
} > "${manifest}"
printf 'Two-GPU formal job submitted; job_id=%s; manifest=%s\n' \
    "${formal_job_id}" "${manifest}"
