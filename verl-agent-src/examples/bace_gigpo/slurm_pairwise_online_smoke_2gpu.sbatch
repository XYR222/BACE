#!/usr/bin/env bash
# C0/C1/C2 online validation from the immutable 2-GPU BACE step-100 checkpoint.
# The default phase is a 2--5 step smoke.  A continuation must be requested
# explicitly with PAIRWISE_PHASE=continuation; it then runs to step 150.
# Submit one independent arm, for example:
#   sbatch --export=ALL,PAIRWISE_ARM=c1 examples/bace_gigpo/slurm_pairwise_online_smoke_2gpu.sbatch
# This launcher is self-contained: it does not invoke another training script.
#SBATCH --job-name=bace-pair-smoke
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
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
SOURCE_STEP=${SOURCE_STEP:-100}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${EXP_ROOT}/preserved_checkpoints/bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry/global_step_${SOURCE_STEP}}
PAIRWISE_ARM=${PAIRWISE_ARM:?Set PAIRWISE_ARM to c0, c1, or c2}
PAIRWISE_PHASE=${PAIRWISE_PHASE:-smoke}
TARGET_STEP=${TARGET_STEP:-102}

[[ "${SOURCE_STEP}" =~ ^[0-9]+$ ]]
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]]
case "${PAIRWISE_PHASE}" in
    smoke)
        (( TARGET_STEP - SOURCE_STEP >= 2 && TARGET_STEP - SOURCE_STEP <= 5 ))
        SAVE_FREQ=1
        MAX_CHECKPOINTS=1
        TEST_FREQ=-1
        VALIDATE_ON_LAST_STEP=false
        ;;
    continuation)
        # The formal comparison is deliberately a three-way fork from the
        # immutable source, never a continuation of a smoke result.
        (( TARGET_STEP == 150 && TARGET_STEP > SOURCE_STEP ))
        SAVE_FREQ=5
        MAX_CHECKPOINTS=2
        TEST_FREQ=5
        VALIDATE_ON_LAST_STEP=true
        ;;
    *)
        printf 'PAIRWISE_PHASE must be smoke or continuation; got %s\n' "${PAIRWISE_PHASE}" >&2
        exit 2
        ;;
esac

case "${PAIRWISE_ARM}" in
    c0)
        PAIRWISE_MODE=full
        MIGRATION_ENABLED=false
        MIGRATION_MODE=strict
        ;;
    c1)
        PAIRWISE_MODE=fixed
        MIGRATION_ENABLED=true
        MIGRATION_MODE=pairwise_from_full
        ;;
    c2)
        PAIRWISE_MODE=stopping
        MIGRATION_ENABLED=true
        MIGRATION_MODE=pairwise_from_full
        ;;
    *)
        printf 'PAIRWISE_ARM must be c0, c1, or c2; got %s\n' "${PAIRWISE_ARM}" >&2
        exit 2
        ;;
esac

RUN_NAME=${BACE_RUN_NAME:-pairwise_${PAIRWISE_ARM}_tau005_from${SOURCE_STEP}_${PAIRWISE_PHASE}_seed0}
INVOCATION_ID=${SLURM_JOB_ID:-local}_${PAIRWISE_ARM}_from${SOURCE_STEP}_to${TARGET_STEP}
ARTIFACT_DIR=${EXP_ROOT}/bace_artifacts/${RUN_NAME}
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
ROLLOUT_DATA_DIR=${EXP_ROOT}/rollout_trajectories/${RUN_NAME}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${INVOCATION_ID}
REPORT_DIR=${EXP_ROOT}/trace_validation/${RUN_NAME}
SMOKE_REPORT=${REPORT_DIR}/${INVOCATION_ID}_pairwise_smoke.json
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${INVOCATION_ID}.log

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WORKSPACE_ROOT}/verl-agent"

# vLLM 0.11.0 on the site TP=2 H100 stack must auto-select FLASH_ATTN;
# pinning XFormers previously failed while building paged KV cache.
unset VLLM_ATTENTION_BACKEND
export VLLM_LOGGING_LEVEL=INFO
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

for path in "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}" \
    "${SOURCE_CHECKPOINT}/.complete" "${SOURCE_CHECKPOINT}/data.pt" \
    "${SOURCE_CHECKPOINT}/bace_collector_state.json"; do
    [[ -e "${path}" ]] || { printf 'Missing required path: %s\n' "${path}" >&2; exit 2; }
done
[[ "$(jq -r '.global_step' "${SOURCE_CHECKPOINT}/bace_collector_state.json")" == "${SOURCE_STEP}" ]]
for rank in 0 1; do
    for component in model optim extra_state; do
        [[ -s "${SOURCE_CHECKPOINT}/actor/${component}_world_size_2_rank_${rank}.pt" ]]
    done
done

mkdir -p "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" \
    "${TENSORBOARD_DIR}" "${METADATA_DIR}" "${REPORT_DIR}" "$(dirname "${LOG_FILE}")"
find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${METADATA_DIR}/source_checkpoint_before.tsv"
cd "${REPO_ROOT}"

python3 "${WORKSPACE_ROOT}/work-BACE/deploy/gpu/check_gpu_environment.py" \
    --expected-gpus 2 --model-path "${MODEL_PATH}" \
    --alfworld-data "${ALFWORLD_DATA}" --output "${METADATA_DIR}/preflight.json"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
python3 -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"

ppo_command=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=bace_gigpo
    algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact
    algorithm.bace.topology=dynamic
    algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed
    algorithm.bace.branch_execution_mode=selected_worker
    algorithm.bace.branch_pool_mode=main_reuse
    algorithm.bace.root_active_executor=true
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
    algorithm.bace.pairwise.mode="${PAIRWISE_MODE}"
    algorithm.bace.pairwise.batch_size=2
    algorithm.bace.checkpoint_migration.enabled="${MIGRATION_ENABLED}"
    algorithm.bace.checkpoint_migration.mode="${MIGRATION_MODE}"
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
    algorithm.bace.artifacts.include_token_arrays=true
    algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true
    algorithm.bace.replay.max_origin_retries=1
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
    "trainer.logger=['console','wandb','tensorboard']"
    trainer.project_name=verl_agent_alfworld
    trainer.experiment_name="${RUN_NAME}"
    trainer.device=cuda
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ}"
    trainer.total_training_steps="${TARGET_STEP}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.resume_mode=resume_path
    trainer.resume_from_path="${SOURCE_CHECKPOINT}"
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs=150
    trainer.val_before_train=False
    trainer.validate_on_last_step="${VALIDATE_ON_LAST_STEP}"
    ray_init.num_cpus=32
)

printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
printf 'arm=%s\nphase=%s\npairwise_mode=%s\nsource_checkpoint=%s\ntarget_step=%s\n' \
    "${PAIRWISE_ARM}" "${PAIRWISE_PHASE}" "${PAIRWISE_MODE}" \
    "${SOURCE_CHECKPOINT}" "${TARGET_STEP}" > "${METADATA_DIR}/arm_contract.txt"
(
    find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo \
        -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
        -print0 | sort -z | xargs -0 sha256sum
) > "${METADATA_DIR}/source_manifest.sha256"

exec > >(tee -a "${LOG_FILE}") 2>&1
"${ppo_command[@]}"

tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
[[ -s "${tracker}" && "$(<"${tracker}")" == "${TARGET_STEP}" ]]
python3 "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" --keep "${MAX_CHECKPOINTS}" \
    --output "${METADATA_DIR}/checkpoint_prune_after.json"
python3 -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${INVOCATION_ID}_trace.json"
python3 "${SCRIPT_DIR}/validate_pairwise_online_smoke.py" "${ARTIFACT_DIR}" \
    --arm "${PAIRWISE_ARM}" --output "${SMOKE_REPORT}"
find "${SOURCE_CHECKPOINT}" -maxdepth 3 -type f \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort > "${METADATA_DIR}/source_checkpoint_after.tsv"
cmp "${METADATA_DIR}/source_checkpoint_before.tsv" "${METADATA_DIR}/source_checkpoint_after.tsv"
