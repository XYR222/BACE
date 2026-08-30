#!/usr/bin/env bash
#SBATCH --job-name=am15b_2g
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=23:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/action-mean-experiments/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/action-mean-experiments/slurm/%x-%j.err

set -euo pipefail

# Self-contained A1/A2 entry point derived from the completed 2-H100 BACE
# launcher. It never invokes another training launcher. ACTION_EXPERIMENT is:
#   A1: pure GiGPO, action-mean local credit, no BACE collector/topology
#   A2: Exact BACE, action-mean local credit (the only algorithm delta vs B0)
ACTION_EXPERIMENT=${ACTION_EXPERIMENT:?set ACTION_EXPERIMENT=A1 or A2}
[[ "${ACTION_EXPERIMENT}" =~ ^A[12]$ ]]
LOCAL_CREDIT_MODE=${LOCAL_CREDIT_MODE:-action_mean}
[[ "${LOCAL_CREDIT_MODE}" =~ ^(occurrence|action_mean)$ ]]
SMOKE=${SMOKE:-0}
[[ "${SMOKE}" =~ ^[01]$ ]]
GPU_COUNT=${GPU_COUNT:-2}
[[ "${GPU_COUNT}" =~ ^[12]$ ]]

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/action-mean-experiments

if [[ "${ACTION_EXPERIMENT}" == A1 ]]; then
    METHOD_SLUG=gigpo
else
    METHOD_SLUG=bace
fi
RUN_KIND=formal
if [[ "${SMOKE}" == 1 ]]; then
    RUN_KIND=smoke
    TARGET_STEP=${TARGET_STEP:-2}
    SAVE_FREQ=${SAVE_FREQ:-2}
else
    TARGET_STEP=${TARGET_STEP:-150}
    SAVE_FREQ=${SAVE_FREQ:-5}
fi
RUN_NAME=${ACTION_RUN_NAME:-${ACTION_EXPERIMENT,,}_${METHOD_SLUG}_${LOCAL_CREDIT_MODE}_${GPU_COUNT}gpu_seed0_${RUN_KIND}}
INVOCATION_ID=${SLURM_JOB_ID:-local}_${ACTION_EXPERIMENT}_${LOCAL_CREDIT_MODE}_to_step_${TARGET_STEP}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-2}

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP >= 1 && TARGET_STEP <= 150 ))
[[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]] && (( SAVE_FREQ >= 1 ))
[[ "${MAX_CHECKPOINTS}" =~ ^(1|2)$ ]]

CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
ROLLOUT_DATA_DIR=${EXP_ROOT}/rollout_trajectories/${RUN_NAME}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION_ID}
REPORT_DIR=${EXP_ROOT}/validation/${RUN_NAME}
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION_ID}.log
ARTIFACT_DIR=${EXP_ROOT}/bace_artifacts/${RUN_NAME}

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
export TOKENIZERS_PARALLELISM=false
export TENSORBOARD_DIR
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
unset PYTORCH_CUDA_ALLOC_CONF

for required_path in "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${required_path}" ]] || {
        printf 'Required path missing: %s\n' "${required_path}" >&2
        exit 2
    }
done

mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" "${TENSORBOARD_DIR}" \
    "${METADATA_DIR}" "${REPORT_DIR}" "$(dirname "${LOG_FILE}")" \
    "${EXP_ROOT}/slurm"
if [[ "${ACTION_EXPERIMENT}" == A2 ]]; then
    mkdir -p "${ARTIFACT_DIR}"
fi
cd "${REPO_ROOT}"

tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
if [[ "${DRY_RUN:-0}" != 1 && -f "${tracker}" ]]; then
    prune_args=()
    [[ "${ACTION_EXPERIMENT}" == A1 ]] && prune_args+=(--allow-missing-bace-state)
    python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep "${MAX_CHECKPOINTS}" \
        "${prune_args[@]}" \
        --output "${METADATA_DIR}/checkpoint_prune_before.json"
    latest_step=$(<"${tracker}")
    if (( latest_step >= TARGET_STEP )); then
        printf 'Target step %s already checkpointed; nothing to resume.\n' "${TARGET_STEP}"
        exit 0
    fi
fi

# All non-method values match the completed 2-H100 reference-aligned run.
ppo_command=(
    python3 -m verl.trainer.main_ppo
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GPU_COUNT}"
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
    algorithm.gigpo.enable_similarity=False
    algorithm.gigpo.compute_mean_std_cross_steps=True
    env.env_name=alfworld/AlfredTWEnv
    env.seed=0
    env.max_steps=50
    env.rollout.n=8
    env.resources_per_worker.num_cpus=0.1
    trainer.critic_warmup=0
    "trainer.logger=['console','tensorboard']"
    trainer.project_name=verl_agent_alfworld_action_mean
    trainer.experiment_name="${RUN_NAME}"
    trainer.device=cuda
    trainer.n_gpus_per_node="${GPU_COUNT}"
    trainer.nnodes=1
    trainer.test_freq=5
    trainer.total_epochs=150
    trainer.val_before_train=True
    trainer.validate_on_last_step=True
    trainer.save_freq="${SAVE_FREQ}"
    trainer.total_training_steps="${TARGET_STEP}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    ray_init.num_cpus="$((GPU_COUNT * 16))"
)

if [[ "${ACTION_EXPERIMENT}" == A1 ]]; then
    ppo_command+=(
        algorithm.adv_estimator=gigpo
        algorithm.bace.enabled=false
        algorithm.gigpo.local_credit_mode="${LOCAL_CREDIT_MODE}"
        algorithm.gigpo.capture_action_metadata=true
    )
else
    ppo_command+=(
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
        algorithm.bace.local_credit_mode="${LOCAL_CREDIT_MODE}"
        algorithm.bace.artifacts.enabled=true
        algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
        algorithm.bace.artifacts.include_token_arrays=true
        algorithm.bace.artifacts.fsync=false
        algorithm.bace.replay.compare_action_set=true
        algorithm.bace.replay.max_origin_retries=1
    )
fi

# One-card, one-hour correctness mode. These bounded execution reductions are
# never used by the formal 2-GPU runs. A2 raises the cold-start readiness only
# in smoke so branch-origin/suffix action identities are exercised immediately.
if [[ "${SMOKE}" == 1 ]]; then
    ppo_command+=(
        data.train_batch_size=2
        data.max_response_length=128
        actor_rollout_ref.actor.ppo_mini_batch_size=16
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
        actor_rollout_ref.actor.use_torch_compile=false
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
        env.max_steps=8
        env.resources_per_worker.num_cpus=0.01
        trainer.test_freq=-1
        trainer.val_before_train=False
        trainer.validate_on_last_step=False
    )
    if [[ "${ACTION_EXPERIMENT}" == A2 ]]; then
        ppo_command+=(
            algorithm.bace.history_initial_mean=0.90
            algorithm.bace.competence_threshold=0.0
        )
    fi
fi

ppo_command+=("$@")
printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
{
    printf 'action_experiment=%s\nlocal_credit_mode=%s\n' \
        "${ACTION_EXPERIMENT}" "${LOCAL_CREDIT_MODE}"
    printf 'run_name=%s\ninvocation_id=%s\ngpu_count=%s\nsmoke=%s\n' \
        "${RUN_NAME}" "${INVOCATION_ID}" "${GPU_COUNT}" "${SMOKE}"
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
    --expected-gpus "${GPU_COUNT}" --model-path "${MODEL_PATH}" \
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
(( train_status == 0 )) || exit "${train_status}"

[[ -f "${tracker}" ]]
[[ "$(<"${tracker}")" == "${TARGET_STEP}" ]]
prune_args=()
[[ "${ACTION_EXPERIMENT}" == A1 ]] && prune_args+=(--allow-missing-bace-state)
python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
    --keep "${MAX_CHECKPOINTS}" \
    "${prune_args[@]}" \
    --output "${METADATA_DIR}/checkpoint_prune_after.json"

if [[ "${ACTION_EXPERIMENT}" == A2 ]]; then
    printf -v final_step_dir '%s/step_%08d' "${ARTIFACT_DIR}" "${TARGET_STEP}"
    python3 -m recipe.bace_gigpo.validate_trace "${final_step_dir}" \
        --output "${REPORT_DIR}/${INVOCATION_ID}_final.json"
fi

# Both A1 and A2 must have emitted the new mechanism metrics.
grep -q 'credit/action_group_count' "${LOG_FILE}"
grep -q 'credit/local_sign_conflict_rate/all' "${LOG_FILE}"
printf 'action_mean_experiment_complete=true\n' >> "${METADATA_DIR}/run_metadata.txt"
