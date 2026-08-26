#!/usr/bin/env bash
#SBATCH --job-name=bace15b_2g_strict_gate
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

# This gate is intentionally self-contained: it does not invoke another
# training launcher or mutate the independent formal script.  The non-BACE
# training parameters below mirror AFH/examples/gigpo_trainer/run_alfworld.sh.
WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
RUN_NAME=selected_worker_2gpu_strict_gate_${SLURM_JOB_ID}
INVOCATION_ID=${SLURM_JOB_ID}_2gpu_strict_gate
ARTIFACT_DIR=${EXP_ROOT}/bace_artifacts/${RUN_NAME}
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION_ID}
REPORT_DIR=${EXP_ROOT}/trace_validation/${RUN_NAME}
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION_ID}.log
FORMAL_SCRIPT=${SCRIPT_DIR}/run_alfworld_h100_2gpu_strict.sbatch

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WORKSPACE_ROOT}/verl-agent"

# Runtime-compatibility exception only: the current vLLM 0.11.0 H100/TP=2
# XFormers path rejects its paged KV-cache block size.  Automatic selection is
# expected to use FLASH_ATTN; the post-run checks below enforce that outcome.
unset VLLM_ATTENTION_BACKEND
# Keep vLLM's INFO-level backend-selection line so the gate can verify the
# actual runtime backend rather than inferring it only from successful startup.
export VLLM_LOGGING_LEVEL=INFO
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
# RWTH's c25g environment can export AMD compatibility selectors alongside
# CUDA_VISIBLE_DEVICES. verl deliberately rejects that ambiguous combination.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

for required_path in \
    "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}" \
    "${FORMAL_SCRIPT}"; do
    [[ -e "${required_path}" ]] || {
        printf 'Required path missing: %s\n' "${required_path}" >&2
        exit 2
    }
done

mkdir -p "${ARTIFACT_DIR}" "${METADATA_DIR}" "${REPORT_DIR}" \
    "$(dirname "${LOG_FILE}")" "${EXP_ROOT}/chain"
cd "${REPO_ROOT}"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | tee "${METADATA_DIR}/gpu_names.txt" | wc -l)
[[ "${gpu_count}" -eq 2 ]] || {
    printf 'Expected exactly 2 visible GPUs, found %s\n' "${gpu_count}" >&2
    exit 3
}
nvidia-smi -q > "${METADATA_DIR}/nvidia_smi_q.txt"

ppo_command=(
    python3 -m verl.trainer.main_ppo

    # BACE method parameters.  Mean 0.60 is gate-only: it makes the first
    # step branch-ready so selected-worker execution is exercised.
    algorithm.adv_estimator=bace_gigpo
    algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact
    algorithm.bace.topology=dynamic
    algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed
    algorithm.bace.branch_execution_mode=selected_worker
    algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.total_leaf_budget=8
    algorithm.bace.min_natural_roots=2
    algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.5
    algorithm.bace.history_initial_mean=0.60
    algorithm.bace.history_initial_strength=2.0
    algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1
    algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0
    algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold=0.005
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity
    algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
    algorithm.bace.artifacts.include_token_arrays=true
    algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true
    algorithm.bace.replay.max_origin_retries=1

    # Full GiGPO/ALFWorld parameter set from the AFH reference script.
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=16
    data.val_batch_size=128
    data.max_prompt_length=2048
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=False
    algorithm.gamma=0.95
    algorithm.gigpo.step_advantage_w=1.0
    algorithm.gigpo.mode=mean_std_norm
    env.env_name=alfworld/AlfredTWEnv
    env.seed=0
    env.max_steps=50
    env.rollout.n=8
    env.resources_per_worker.num_cpus=0.1
    trainer.critic_warmup=0
    "trainer.logger=['console','wandb']"
    trainer.project_name=verl_agent_alfworld
    trainer.experiment_name="${RUN_NAME}"
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=150
    trainer.val_before_train=True

    # Gate control only; all optimizer/rollout parameters above stay intact.
    trainer.total_training_steps=1
)

printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
printf 'VLLM_ATTENTION_BACKEND=<unset>; policy=automatic; expected=FLASH_ATTN\n' \
    > "${METADATA_DIR}/attention_backend_policy.txt"

set -o pipefail
"${ppo_command[@]}" 2>&1 | tee -a "${LOG_FILE}"

# The process must have completed the PPO update, not merely written rollout
# artifacts before a later OOM.
grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" \
    "${LOG_FILE}"
# The multiprocess resource tracker can print an "Exception ignored" traceback
# from __del__ after a successful Python exit.  The pipeline exit status above
# already proves main_ppo returned zero, so reject concrete training failures
# without treating that known interpreter-shutdown warning as a failed step.
! grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|Error executing job with overrides|ray\.exceptions\.RayTaskError' \
    "${LOG_FILE}"

# Unsetting the override is not sufficient evidence by itself: prove that
# vLLM selected FlashAttention and did not silently fall back to XFormers.
grep -Eqi 'Using (the )?FLASH_ATTN backend|Using Flash Attention backend|FlashAttention[^[:space:]]*.*backend' \
    "${LOG_FILE}"
! grep -Eqi 'Using (the )?XFORMERS backend|XFormersAttentionMetadataBuilder' \
    "${LOG_FILE}"

resolved_command=${METADATA_DIR}/resolved_command.sh
grep -Fq 'actor_rollout_ref.rollout.tensor_model_parallel_size=2' "${resolved_command}"
grep -Fq 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32' "${resolved_command}"
grep -Fq 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32' "${resolved_command}"
grep -Fq 'algorithm.bace.variant=batch_erv_exact' "${resolved_command}"
grep -Fq 'algorithm.bace.staged_root_batching=packed' "${resolved_command}"
grep -Fq 'algorithm.bace.branch_execution_mode=selected_worker' "${resolved_command}"
grep -Fq 'trainer.total_training_steps=1' "${resolved_command}"

python3 -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${INVOCATION_ID}_final.json"
python3 "${SCRIPT_DIR}/validate_selected_worker_metrics.py" "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${INVOCATION_ID}_selected_worker.json"

python3 - "${ARTIFACT_DIR}/step_00000001" <<'PY'
import json
import sys
from pathlib import Path

step_dir = Path(sys.argv[1])
summary = json.loads((step_dir / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads((step_dir / "manifest.json").read_text(encoding="utf-8"))
metrics = summary.get("bace_metrics", {})
assert summary.get("status") == "complete"
assert manifest.get("variant") == "batch_erv_exact"
assert manifest.get("staged_root_batching") == "packed"
assert manifest.get("branch_execution_mode") == "selected_worker"
assert float(metrics.get("batch_erv_exact", 0)) == 1.0
assert float(metrics.get("staged_root_batching_packed", 0)) == 1.0
assert float(metrics.get("branch_selected_worker_execution", 0)) == 1.0
assert int(metrics.get("requested", 0)) > 0
assert int(metrics.get("validated", -1)) == int(metrics["requested"])
print("Exact/packed/selected-worker branch gate passed")
PY

# This line is reachable only when training and every gate above succeeded.
# The first formal job targets step 150 directly and checkpoints every 5 steps.
# A single afternotok recovery allocation resumes its latest committed step if
# the first allocation times out or fails; neither job has a fixed segment end.
formal_job_id=$(sbatch --parsable "${FORMAL_SCRIPT}")
recovery_job_id=$(sbatch --parsable --dependency="afternotok:${formal_job_id}" \
    "${FORMAL_SCRIPT}")
manifest_file=${EXP_ROOT}/chain/two_gpu_strict_from_gate_${SLURM_JOB_ID}.txt
{
    printf 'gate_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'gate_run_name=%s\n' "${RUN_NAME}"
    printf 'gate_status=passed\n'
    printf 'gate_actor_micro_batch=32\n'
    printf 'gate_logprob_micro_batch=32\n'
    printf 'gate_tensor_parallel_size=2\n'
    printf 'gate_attention_backend=FLASH_ATTN(auto)\n'
    printf 'formal_script=%s\n' "${FORMAL_SCRIPT}"
    printf 'formal_job_id=%s\n' "${formal_job_id}"
    printf 'recovery_dependency=afternotok:%s\n' "${formal_job_id}"
    printf 'recovery_job_id=%s\n' "${recovery_job_id}"
    printf 'checkpoint_save_freq=5\n'
    printf 'checkpoint_keep=1\n'
    printf 'target_step=150\n'
} > "${manifest_file}"
printf 'All gates passed; submitted two-GPU formal job %s and recovery job %s\n' \
    "${formal_job_id}" "${recovery_job_id}"
