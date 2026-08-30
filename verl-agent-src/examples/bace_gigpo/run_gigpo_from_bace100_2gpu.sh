#!/usr/bin/env bash
set -euo pipefail

# Causal method fork: restore every step-coupled training state from the
# two-GPU BACE step-100 checkpoint, but use the native GiGPO collector and
# advantage estimator from step 101 onward.  The source directory is read-only;
# all new checkpoints and logs live under a distinct run name.
WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${EXP_ROOT}/preserved_checkpoints/bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry/global_step_100}
RUN_NAME=${GIGPO_FORK_RUN_NAME:-gigpo_from_bace2gpu_step100_seed0}
TARGET_STEP=${TARGET_STEP:-150}
SAVE_FREQ=${SAVE_FREQ:-5}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-2}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
VALIDATE_ON_LAST_STEP=${VALIDATE_ON_LAST_STEP:-true}
INVOCATION_ID=${GIGPO_FORK_INVOCATION_ID:-${SLURM_JOB_ID:-local}_to_step_${TARGET_STEP}}

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP >= 101 && TARGET_STEP <= 150 ))
[[ "${SAVE_FREQ}" =~ ^-?[0-9]+$ ]]
[[ "${MAX_CHECKPOINTS}" =~ ^(1|2)$ ]]

CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
ROLLOUT_DATA_DIR=${EXP_ROOT}/rollout_trajectories/${RUN_NAME}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION_ID}
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION_ID}.log
VALIDATION_DATA_DIR=${EXP_ROOT}/validation/${RUN_NAME}
SOURCE_REPORT=${METADATA_DIR}/source_checkpoint_validation.json

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WORKSPACE_ROOT}/verl-agent"

unset VLLM_ATTENTION_BACKEND
export VLLM_LOGGING_LEVEL=INFO
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TENSORBOARD_DIR
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
unset PYTORCH_CUDA_ALLOC_CONF

for required_path in "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${required_path}" ]] || { printf 'Required path missing: %s\n' "${required_path}" >&2; exit 2; }
done
mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" "${TENSORBOARD_DIR}" \
    "${METADATA_DIR}" "$(dirname "${LOG_FILE}")" "${VALIDATION_DATA_DIR}"
cd "${REPO_ROOT}"

python3 "${SCRIPT_DIR}/validate_bace100_fork_source.py" \
    "${SOURCE_CHECKPOINT}" --expected-step 100 --output "${SOURCE_REPORT}"

# Refuse accidental in-place continuation: a GiGPO fork must never rotate or
# otherwise mutate the BACE source checkpoint tree.
source_root=$(realpath "$(dirname "${SOURCE_CHECKPOINT}")")
destination_root=$(realpath -m "${CHECKPOINT_DIR}")
[[ "${destination_root}" != "${source_root}" && "${destination_root}" != "${source_root}/"* ]] || {
    printf 'GiGPO destination overlaps the BACE source root.\n' >&2
    exit 2
}

ppo_command=(
    python3 -m verl.trainer.main_ppo

    # The only algorithmic intervention relative to the source run.
    algorithm.adv_estimator=gigpo
    algorithm.bace.enabled=false

    # Training and rollout settings match the two-GPU BACE source run.
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
    "trainer.logger=['console','tensorboard']"
    trainer.project_name=verl_agent_alfworld
    trainer.experiment_name="${RUN_NAME}"
    trainer.device=cuda
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs=150
    trainer.total_training_steps="${TARGET_STEP}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.validate_on_last_step="${VALIDATE_ON_LAST_STEP}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.resume_mode=resume_path
    trainer.resume_from_path="${SOURCE_CHECKPOINT}"
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-32}"
)
ppo_command+=("$@")

printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
{
    printf 'fork_semantics=restore_actor_optimizer_scheduler_rng_dataloader;replace_bace_with_gigpo\n'
    printf 'source_checkpoint=%q\ndestination_checkpoint_root=%q\n' "${SOURCE_CHECKPOINT}" "${CHECKPOINT_DIR}"
    printf 'run_name=%q\ninvocation_id=%q\nstart_time=%s\n' \
        "${RUN_NAME}" "${INVOCATION_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${METADATA_DIR}/run_metadata.txt"
find verl/trainer agent_system examples/bace_gigpo -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
    -print0 | sort -z | xargs -0 sha256sum > "${METADATA_DIR}/source_manifest.sha256"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN=1; resolved command written to %s\n' "${METADATA_DIR}/resolved_command.sh"
    exit 0
fi

python3 "${WORKSPACE_ROOT}/work-BACE/deploy/gpu/check_gpu_environment.py" \
    --expected-gpus 2 --model-path "${MODEL_PATH}" --alfworld-data "${ALFWORLD_DATA}" \
    --output "${METADATA_DIR}/preflight.json"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
python3 -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"

exec > >(tee -a "${LOG_FILE}") 2>&1
set +e
"${ppo_command[@]}"
train_status=$?
set -e
printf 'end_time=%s\ntrain_exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${train_status}" >> "${METADATA_DIR}/run_metadata.txt"
(( train_status == 0 )) || exit "${train_status}"

tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
[[ -f "${tracker}" && "$(<"${tracker}")" == "${TARGET_STEP}" ]]
[[ -f "${CHECKPOINT_DIR}/global_step_${TARGET_STEP}/actor/model_world_size_2_rank_0.pt" ]]
[[ -f "${CHECKPOINT_DIR}/global_step_${TARGET_STEP}/actor/model_world_size_2_rank_1.pt" ]]
python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
    --keep "${MAX_CHECKPOINTS}" --allow-missing-bace-state \
    --output "${METADATA_DIR}/checkpoint_prune_after.json"
