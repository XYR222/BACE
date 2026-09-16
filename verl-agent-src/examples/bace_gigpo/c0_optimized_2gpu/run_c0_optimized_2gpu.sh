#!/usr/bin/env bash
# Standalone C0/P1-S 2xH100 launcher.  It never sources or execs another launcher.
#SBATCH --job-name=bace_c0_opt
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=23:30:00
#SBATCH --chdir=.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Accept only a complete veRL repository, not an arbitrary submit directory.
if [[ -n "${BACE_REPO_ROOT:-}" ]]; then
    REPO_ROOT=$(cd "${BACE_REPO_ROOT}" && pwd -P)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/verl/trainer/main_ppo.py" ]]; then
    REPO_ROOT=$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)
else
    REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd -P)
fi
[[ -f "${REPO_ROOT}/verl/trainer/main_ppo.py" ]] || { echo "Invalid veRL repository: ${REPO_ROOT}" >&2; exit 2; }
WORK_BACE_ROOT=$(cd "${REPO_ROOT}/.." && pwd)
WORKSPACE_ROOT=$(cd "${WORK_BACE_ROOT}/.." && pwd)
EXP_ROOT=${WORK_BACE_ROOT}/experiments/alfworld-qwen2.5-1.5b-exact
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet

# Grid axes.  These are the only three method parameters intentionally swept.
SEED=${BACE_SEED:-0}
COMPETENCE_THRESHOLD=${C0_COMPETENCE_THRESHOLD:-0.5}
STEP_ADVANTAGE_W=${C0_STEP_ADVANTAGE_W:-1.0}
BATCH_ERV_THRESHOLD=${C0_BATCH_ERV_THRESHOLD:-0.005}
# Keep the historical optimized-C0 correction protocol fixed.  This is not a
# sweep axis: a correction wave may convert at most one still-planned branch
# slot back to a natural root for a task.  Raising it to four is a distinct
# root-generation schedule and must not be compared as the old C0 baseline.
CAPACITY_CORRECTION_BATCH_SIZE=1
TARGET_STEP=${TARGET_STEP:-150}
RUN_NAME=${BACE_RUN_NAME:-bace_c0_opt_s${SEED}_c${COMPETENCE_THRESHOLD}_w${STEP_ADVANTAGE_W}_tau${BATCH_ERV_THRESHOLD}}
SAVE_FREQ=${SAVE_FREQ:-5}
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-2}
MILESTONE_CHECKPOINT_STEPS=${MILESTONE_CHECKPOINT_STEPS:-10,75,100,140}
GPU_COUNT=${BACE_GPU_COUNT:-2}
ROLLOUT_TP=${BACE_ROLLOUT_TP:-${GPU_COUNT}}
ACTOR_MICRO_BATCH=${BACE_ACTOR_MICRO_BATCH:-32}
LOGPROB_MICRO_BATCH=${BACE_LOGPROB_MICRO_BATCH:-32}
ROLLOUT_GPU_MEMORY_UTILIZATION=${BACE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}

[[ "${SEED}" =~ ^[0-9]+$ ]]
[[ "${COMPETENCE_THRESHOLD}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]
[[ "${STEP_ADVANTAGE_W}" =~ ^[0-9]+(\.[0-9]+)?$ ]]
[[ "${BATCH_ERV_THRESHOLD}" =~ ^0\.[0-9]+$ ]]
[[ "${TARGET_STEP}" =~ ^[0-9]+$ ]] && (( TARGET_STEP >= 1 && TARGET_STEP <= 150 ))
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${MAX_CHECKPOINTS}" =~ ^(1|2)$ ]]
[[ "${GPU_COUNT}" =~ ^[12]$ && "${ROLLOUT_TP}" =~ ^[12]$ ]]
(( ROLLOUT_TP <= GPU_COUNT ))
[[ "${ACTOR_MICRO_BATCH}" =~ ^[1-9][0-9]*$ && "${LOGPROB_MICRO_BATCH}" =~ ^[1-9][0-9]*$ ]]
if [[ "${MILESTONE_CHECKPOINT_STEPS}" == none ]]; then
    MILESTONE_ENABLED=false
else
    MILESTONE_ENABLED=true
    [[ "${MILESTONE_CHECKPOINT_STEPS}" =~ ^[0-9]+(,[0-9]+)*$ ]]
fi

ARTIFACT_DIR=${EXP_ROOT}/bace_artifacts/${RUN_NAME}
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
ROLLOUT_DIR=${EXP_ROOT}/rollout_trajectories/${RUN_NAME}
TB_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
export TENSORBOARD_DIR=${TB_DIR}
WANDB_ROOT=${EXP_ROOT}/wandb/${RUN_NAME}
export WANDB_DIR=${WANDB_ROOT}/runs
export WANDB_DATA_DIR=${WANDB_ROOT}/data
export WANDB_CACHE_DIR=${WANDB_ROOT}/cache
export WANDB_ARTIFACT_DIR=${WANDB_ROOT}/artifacts
METADATA_DIR=${EXP_ROOT}/run_metadata/${RUN_NAME}/${SLURM_JOB_ID:-local}
REPORT_DIR=${EXP_ROOT}/trace_validation/${RUN_NAME}
LOG_FILE=${EXP_ROOT}/logs/${RUN_NAME}/${SLURM_JOB_ID:-local}.log
MILESTONE_DIR=${EXP_ROOT}/preserved_checkpoints/${RUN_NAME}

CONDA_BASE=${CONDA_BASE:-${HOME}/miniforge3}
VERL_AGENT_ENV=${WORKSPACE_ROOT}/verl-agent
module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${VERL_AGENT_ENV}"
unset VLLM_ATTENTION_BACKEND
export VLLM_LOGGING_LEVEL=INFO ALFWORLD_DATA=${ALFWORLD_DATA:-${HOME}/.cache/alfworld}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# TensorBoard uses a relative tensorboard_log directory.  Enter the repository,
# where that path is linked to the rwth2089 experiment storage volume.
cd "${REPO_ROOT}"
for path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${ALFWORLD_DATA}"; do [[ -e "${path}" ]] || { echo "Missing ${path}" >&2; exit 2; }; done
mkdir -p "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${TB_DIR}" \
    "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" \
    "${METADATA_DIR}" "${REPORT_DIR}" "$(dirname "${LOG_FILE}")"
[[ "${MILESTONE_ENABLED}" == true ]] && mkdir -p "${MILESTONE_DIR}"

# Independent node-local Ray state: safe if two 2-GPU jobs share a 4-GPU node.
RAY_TMP=/tmp/bace_ray_${USER}_${SLURM_JOB_ID:-local_c0}
mkdir -p "${RAY_TMP}/ray" "${RAY_TMP}/hf_datasets"
export TMPDIR=${RAY_TMP} RAY_TMPDIR=${RAY_TMP} HF_DATASETS_CACHE=${RAY_TMP}/hf_datasets

tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
if [[ "${DRY_RUN:-0}" != 1 && -f "${tracker}" ]]; then
    python3 ${REPO_ROOT}/examples/bace_gigpo/prune_checkpoints.py "${CHECKPOINT_DIR}" --keep "${MAX_CHECKPOINTS}" --output "${METADATA_DIR}/checkpoint_prune_before.json"
fi

cmd=(python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=bace_gigpo
    algorithm.bace.enabled=true algorithm.bace.variant=batch_erv_exact algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.topology=dynamic algorithm.bace.dynamic_root_generation=staged algorithm.bace.staged_root_batching=packed
    algorithm.bace.branch_execution_mode=selected_worker algorithm.bace.branch_pool_mode=main_reuse algorithm.bace.root_active_executor=true
    algorithm.bace.total_leaf_budget=8 algorithm.bace.min_natural_roots=2 algorithm.bace.capacity_correction_batch_size="${CAPACITY_CORRECTION_BATCH_SIZE}" algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold="${COMPETENCE_THRESHOLD}"
    algorithm.bace.history_initial_mean=0.10 algorithm.bace.history_initial_strength=2.0 algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1 algorithm.bace.history_min_strength=2.0 algorithm.bace.history_max_strength=8.0 algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold="${BATCH_ERV_THRESHOLD}" algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12 algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity algorithm.bace.tie_break_identity_mode=stable_v1 algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.credit_mode=current algorithm.bace.tree_credit_mode=current algorithm.bace.macro_normalization_mode=stable_occurrence algorithm.bace.tree_ppo_padding_mode=copy_trainable
    algorithm.bace.artifacts.enabled=true algorithm.bace.artifacts.directory="${ARTIFACT_DIR}" algorithm.bace.artifacts.include_token_arrays=true algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true algorithm.bace.replay.max_origin_retries=1
    data.train_files="${TRAIN_FILE}" data.val_files="${VAL_FILE}" data.train_batch_size=16 data.val_batch_size=128 data.max_prompt_length=2048 data.max_response_length=512 data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True
    actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.use_torch_compile=True actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.max_num_seqs=1024 actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=False algorithm.gamma=0.95 algorithm.gigpo.step_advantage_w="${STEP_ADVANTAGE_W}" algorithm.gigpo.mode=mean_std_norm
    env.env_name=alfworld/AlfredTWEnv env.seed="${SEED}" env.max_steps=50 env.rollout.n=8 env.resources_per_worker.num_cpus=0.1
    trainer.critic_warmup=0 "trainer.logger=['console','wandb','tensorboard']" trainer.project_name=verl_agent_alfworld trainer.experiment_name="${RUN_NAME}" trainer.device=cuda trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.test_freq=5 trainer.total_epochs=150 trainer.val_before_train=True trainer.validate_on_last_step=True
    trainer.save_freq="${SAVE_FREQ}" trainer.total_training_steps="${TARGET_STEP}" trainer.default_local_dir="${CHECKPOINT_DIR}" trainer.rollout_data_dir="${ROLLOUT_DIR}" trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    ray_init.num_cpus=32 +ray_init._temp_dir="${RAY_TMP}/ray" +ray_init.include_dashboard=False)
[[ "${MILESTONE_ENABLED}" == true ]] && cmd+=("trainer.milestone_checkpoint_steps=[${MILESTONE_CHECKPOINT_STEPS}]" trainer.milestone_checkpoint_dir="${MILESTONE_DIR}")
if [[ "${GPU_COUNT}" != 2 || "${ROLLOUT_TP}" != 2 || "${ACTOR_MICRO_BATCH}" != 32 || "${LOGPROB_MICRO_BATCH}" != 32 || "${ROLLOUT_GPU_MEMORY_UTILIZATION}" != 0.6 ]]; then
    cmd+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH}" actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH}" actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH}" actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" trainer.n_gpus_per_node="${GPU_COUNT}")
fi

printf '%q ' "${cmd[@]}" > "${METADATA_DIR}/resolved_command.sh"; printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
printf 'method=C0\nseed=%s\ncompetence_threshold=%s\nstep_advantage_w=%s\nbatch_erv_threshold=%s\ncapacity_correction_batch_size=%s\nscheduling_profile=optimized\ncredit_mode=current\n' "${SEED}" "${COMPETENCE_THRESHOLD}" "${STEP_ADVANTAGE_W}" "${BATCH_ERV_THRESHOLD}" "${CAPACITY_CORRECTION_BATCH_SIZE}" > "${METADATA_DIR}/run_metadata.txt"
(cd "${REPO_ROOT}" && find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) -print0 | sort -z | xargs -0 sha256sum) > "${METADATA_DIR}/source_manifest.sha256"

[[ "${DRY_RUN:-0}" == 1 ]] && { echo "DRY_RUN: ${METADATA_DIR}/resolved_command.sh"; exit 0; }
python3 ${WORKSPACE_ROOT}/work-BACE/deploy/gpu/check_gpu_environment.py --expected-gpus "${GPU_COUNT}" --model-path "${MODEL_PATH}" --alfworld-data "${ALFWORLD_DATA}" --output "${METADATA_DIR}/preflight.json"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
set +e; "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"; status=${PIPESTATUS[0]}; set -e
printf 'train_exit_code=%s\n' "${status}" >> "${METADATA_DIR}/run_metadata.txt"
if find "${ARTIFACT_DIR}" -name summary.json -print -quit | grep -q .; then python3 -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" --output "${REPORT_DIR}/${SLURM_JOB_ID:-local}_final.json"; fi
[[ -f "${tracker}" ]] && python3 ${REPO_ROOT}/examples/bace_gigpo/prune_checkpoints.py "${CHECKPOINT_DIR}" --keep "${MAX_CHECKPOINTS}" --output "${METADATA_DIR}/checkpoint_prune_after.json"
exit "${status}"
