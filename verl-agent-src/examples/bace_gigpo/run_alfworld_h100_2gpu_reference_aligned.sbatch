#!/usr/bin/env bash
#SBATCH --job-name=bace15b_2g_ref
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=23:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

# Self-contained 2 x H100 BACE entry point.  It deliberately does not source
# or invoke the 1/4-GPU BACE launchers.  Non-BACE training parameters match
# AFH/examples/gigpo_trainer/run_alfworld.sh, with inherited defaults written
# explicitly so a future config-default change cannot silently alter the run.
WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact

RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_qwen2_5_1_5b_exact_2gpu_reference_seed0}
INVOCATION_ID=${SLURM_JOB_ID:-local}_2gpu_reference_to_step_${TARGET_STEP:-150}
TARGET_STEP=${TARGET_STEP:-150}
SAVE_FREQ=${SAVE_FREQ:-5}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-1}
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP >= 1 && TARGET_STEP <= 150 ))
[[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]] && (( SAVE_FREQ >= 1 ))
[[ "${MAX_CHECKPOINTS}" =~ ^(1|2)$ ]]

ARTIFACT_DIR=${EXP_ROOT}/bace_artifacts/${RUN_NAME}
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
ROLLOUT_DATA_DIR=${EXP_ROOT}/rollout_trajectories/${RUN_NAME}
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION_ID}
REPORT_DIR=${EXP_ROOT}/trace_validation/${RUN_NAME}
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION_ID}.log

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WORKSPACE_ROOT}/verl-agent"

# AFH pins XFORMERS, but vLLM 0.11.0's V1 XFormers TP=2 decode path failed
# while constructing the paged KV cache on the cluster H100 stack.  Backend
# auto-selection chooses FLASH_ATTN and does not change a training hyperparameter.
unset VLLM_ATTENTION_BACKEND
export VLLM_LOGGING_LEVEL=INFO
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# Bound native thread pools inside the Slurm cgroup.  This is operational only
# and prevents the process/thread growth observed in earlier long jobs.
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
unset PYTORCH_CUDA_ALLOC_CONF

for required_path in \
    "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${required_path}" ]] || {
        printf 'Required path missing: %s\n' "${required_path}" >&2
        exit 2
    }
done

mkdir -p "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" \
    "${METADATA_DIR}" "${REPORT_DIR}" "$(dirname "${LOG_FILE}")"
cd "${REPO_ROOT}"

# Recognize and rotate checkpoints left by a previous Slurm process before the
# trainer creates its in-memory checkpoint manager.
tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
if [[ "${DRY_RUN:-0}" != 1 && -f "${tracker}" ]]; then
    python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep "${MAX_CHECKPOINTS}" \
        --output "${METADATA_DIR}/checkpoint_prune_before.json"
    latest_step=$(<"${tracker}")
    if (( latest_step >= TARGET_STEP )); then
        python3 -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
            --output "${REPORT_DIR}/${INVOCATION_ID}_already_complete.json"
        printf 'Target step %s already checkpointed; nothing to resume.\n' "${TARGET_STEP}"
        exit 0
    fi
fi

ppo_command=(
    python3 -m verl.trainer.main_ppo

    # BACE-only method configuration.
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
    algorithm.bace.history_initial_mean=0.10
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
    algorithm.bace.tie_break_identity_mode=stable_v1
    algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
    algorithm.bace.artifacts.include_token_arrays=true
    algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true
    algorithm.bace.replay.max_origin_retries=1

    # Effective GiGPO reference parameters.
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
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=1024
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
    trainer.device=cuda
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.test_freq=5
    trainer.total_epochs=150
    trainer.val_before_train=True
    trainer.validate_on_last_step=True

    # Wall-time/checkpoint operations; no optimizer or rollout math changes.
    trainer.save_freq="${SAVE_FREQ}"
    trainer.total_training_steps="${TARGET_STEP}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    ray_init.num_cpus=32
)

# Optional Hydra overrides are appended last for a bounded smoke/debug run.
ppo_command+=("$@")
printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"

{
    printf 'run_name=%q\ninvocation_id=%q\n' "${RUN_NAME}" "${INVOCATION_ID}"
    printf 'model_path=%q\ntrain_file=%q\nval_file=%q\n' \
        "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"
    printf 'reference_script=%q\n' \
        "${WORKSPACE_ROOT}/AFH/examples/gigpo_trainer/run_alfworld.sh"
    printf 'attention_backend=automatic_expected_FLASH_ATTN\n'
    printf 'start_time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${METADATA_DIR}/run_metadata.txt"

(
    find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo \
        -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
        -print0 | sort -z | xargs -0 sha256sum
) > "${METADATA_DIR}/source_manifest.sha256"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN=1; resolved command written to %s\n' \
        "${METADATA_DIR}/resolved_command.sh"
    exit 0
fi

python3 "${WORKSPACE_ROOT}/work-BACE/deploy/gpu/check_gpu_environment.py" \
    --expected-gpus 2 --model-path "${MODEL_PATH}" \
    --alfworld-data "${ALFWORLD_DATA}" \
    --output "${METADATA_DIR}/preflight.json"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
python3 -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"

exec > >(tee -a "${LOG_FILE}") 2>&1
set +e
"${ppo_command[@]}"
train_status=$?
set -e
printf 'end_time=%s\ntrain_exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${train_status}" \
    >> "${METADATA_DIR}/run_metadata.txt"

if find "${ARTIFACT_DIR}" -name summary.json -print -quit | grep -q .; then
    python3 -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
        --output "${REPORT_DIR}/${INVOCATION_ID}_final.json"
fi
if [[ -f "${tracker}" ]]; then
    python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep "${MAX_CHECKPOINTS}" \
        --output "${METADATA_DIR}/checkpoint_prune_after.json"
fi
exit "${train_status}"
