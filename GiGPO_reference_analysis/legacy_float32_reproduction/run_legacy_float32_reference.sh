#!/usr/bin/env bash
# Frozen, legacy-float32 GiGPO reproduction launcher.
#
# Training code is imported exclusively from the seed-0 source archive.  The
# only intentional training-time difference from seed 0 is REF_SEED.  Recorder
# milestone snapshots are disabled because each full optimizer snapshot is
# ~19 GiB; this changes neither rollout nor PPO updates and keeps the two
# reproductions within the current filesystem capacity.

set -euo pipefail

PROJECT_ROOT=/hpcwork/xsz96350/fu_project
ARCHIVE_ROOT=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-reference
LEGACY_SOURCE="${ARCHIVE_ROOT}/frozen_sources/seed0_float32_legacy"
LEGACY_SOURCE_ARCHIVE="${ARCHIVE_ROOT}/runs/seed_0_v2/manifest/source_snapshot.tar.gz"
LEGACY_ARCHIVE_SHA=935850afc22d853604f8e54087b77749815071754e0589e703e5a3de2fa36644
LEGACY_CORE_SHA=a9057e4ec40dcf317b4e3733e377be3b08139910e1c8e8d019caec6ddf85c6c3
MODEL_PATH="${PROJECT_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct"
TRAIN_FILE="${PROJECT_ROOT}/data/text/train.parquet"
VAL_FILE="${PROJECT_ROOT}/data/text/test.parquet"
ALFWORLD_ROOT=/home/xsz96350/.cache/alfworld

: "${REF_SEED:?REF_SEED must be supplied by the Slurm wrapper}"
RUN_ID="${REF_RUN_ID:-gigpo_ref_v2_seed${REF_SEED}_legacy_float32}"
RUN_ROOT="${REF_RUN_ROOT:-${ARCHIVE_ROOT}/runs/seed_${REF_SEED}_legacy_float32_v1}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/live"
INVOCATION_DIR="${RUN_ROOT}/segments/${SLURM_JOB_ID:-manual}"

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${PROJECT_ROOT}/verl-agent"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=INFO
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export ALFWORLD_DATA="${ALFWORLD_ROOT}"
# Must precede the editable AFH install, otherwise the repaired current source
# would be imported instead of the frozen legacy source.
export PYTHONPATH="${LEGACY_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

for required in "${LEGACY_SOURCE}" "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${ALFWORLD_ROOT}"; do
    [[ -e "${required}" ]] || { printf 'Missing required asset: %s\n' "${required}" >&2; exit 2; }
done
[[ -f "${MODEL_PATH}/model.safetensors.index.json" || -f "${MODEL_PATH}/model.safetensors" ]] || {
    printf 'No supported safetensors weights found in %s\n' "${MODEL_PATH}" >&2
    exit 2
}

archive_sha=$(sha256sum "${LEGACY_SOURCE_ARCHIVE}" | awk '{print $1}')
core_sha=$(sha256sum "${LEGACY_SOURCE}/gigpo/core_gigpo.py" | awk '{print $1}')
[[ "${archive_sha}" == "${LEGACY_ARCHIVE_SHA}" ]] || { echo 'legacy source archive checksum mismatch' >&2; exit 2; }
[[ "${core_sha}" == "${LEGACY_CORE_SHA}" ]] || { echo 'legacy core_gigpo checksum mismatch' >&2; exit 2; }

mkdir -p "${INVOCATION_DIR}" "${CHECKPOINT_DIR}" "${RUN_ROOT}/manifest" "${RUN_ROOT}/integrity" "${RUN_ROOT}/profiler"
export WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"
if compgen -G "${RUN_ROOT}/integrity/updates/update_*.complete.json" >/dev/null; then
    printf 'Legacy reference run %s already contains completed updates; refusing to mix runs.\n' "${RUN_ID}" >&2
    exit 4
fi

# A run retains only the trainer's rolling recovery checkpoint.  The reference
# recorder's 15 milestone copies are intentionally disabled for capacity.
available_kib=$(df -Pk "${ARCHIVE_ROOT}" | awk 'NR==2 {print $4}')
(( available_kib >= 80 * 1024 * 1024 )) || { printf 'Insufficient free space: %s KiB\n' "${available_kib}" >&2; exit 3; }

ppo_command=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=gigpo
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
    env.seed="${REF_SEED}"
    env.max_steps=50
    env.rollout.n=8
    env.resources_per_worker.num_cpus=0.1
    trainer.critic_warmup=0
    "trainer.logger=['console','wandb']"
    trainer.project_name=verl_agent_alfworld_reference
    trainer.experiment_name="${RUN_ID}"
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq=5
    trainer.max_actor_ckpt_to_keep=2
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=disable
    trainer.test_freq=5
    trainer.total_epochs=150
    trainer.val_before_train=True
    trainer.reference_recorder.enabled=true
    trainer.reference_recorder.schema_version=gigpo_alfworld_ref_v2
    trainer.reference_recorder.output_dir="${RUN_ROOT}"
    trainer.reference_recorder.run_id="${RUN_ID}"
    trainer.reference_recorder.record_tokens=true
    trainer.reference_recorder.record_logprobs=true
    trainer.reference_recorder.record_validation=true
    trainer.reference_recorder.record_profiler=true
    trainer.reference_recorder.checksum_shards=true
    trainer.reference_recorder.actor_snapshot_freq=0
    trainer.reference_recorder.resumable_milestone_freq=0
)

printf '%q ' "${ppo_command[@]}" > "${INVOCATION_DIR}/resolved_command.sh"
printf '\n' >> "${INVOCATION_DIR}/resolved_command.sh"
python3 - <<'PY' > "${INVOCATION_DIR}/legacy_import_check.txt"
import gigpo.core_gigpo as core
assert not hasattr(core, "_stable_group_normalize"), core.__file__
print(core.__file__)
print("legacy_float32_import=ok")
PY

(
    cd "${LEGACY_SOURCE}"
    "${ppo_command[@]}" --cfg job --resolve \
        > "${INVOCATION_DIR}/resolved_config.yaml" \
        2> "${INVOCATION_DIR}/resolved_config.stderr"
)

# Keep an exact, hard-linked copy of the immutable seed-0 source archive.
# The frozen directory adds only the packaging-generated version/version shim;
# its core file hash is recorded below and checked before every invocation.
ln "${LEGACY_SOURCE_ARCHIVE}" "${RUN_ROOT}/manifest/seed0_source_snapshot.tar.gz"
printf '{"legacy_source_archive_sha256":"%s","legacy_core_gigpo_sha256":"%s","normalization":"legacy_float32_mean_std_norm","recorder_milestones":false}\n' \
    "${archive_sha}" "${core_sha}" > "${RUN_ROOT}/manifest/legacy_provenance.json"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Legacy float32 dry-run complete: %s\n' "${INVOCATION_DIR}/resolved_command.sh"
    exit 0
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 5 \
    > "${RUN_ROOT}/profiler/gpu_${SLURM_JOB_ID:-manual}.csv" 2>&1 &
profiler_pid=$!
trap 'kill "${profiler_pid}" 2>/dev/null || true' EXIT
(
    cd "${LEGACY_SOURCE}"
    "${ppo_command[@]}"
) 2>&1 | tee "${INVOCATION_DIR}/trainer.log"
kill "${profiler_pid}" 2>/dev/null || true
wait "${profiler_pid}" 2>/dev/null || true
trap - EXIT

# The frozen legacy validator is preserved as an audit, not a success gate:
# its historical float32-vs-float64 mismatch is the reason for this rerun.
set +e
(
    cd "${LEGACY_SOURCE}"
    python3 examples/gigpo_trainer/validate_reference_archive.py "${RUN_ROOT}" \
        --expected-updates 150 --expected-validations 31 \
        --output "${RUN_ROOT}/integrity/final_validation_legacy.json"
)
archive_validation_exit=$?
(
    cd "${LEGACY_SOURCE}"
    python3 examples/gigpo_trainer/validate_reference_replay.py "${RUN_ROOT}" \
        --samples 32 --seed "${REF_SEED}" \
        --output "${RUN_ROOT}/integrity/prefix_replay_validation.json"
)
replay_validation_exit=$?
set -e
printf '{"status":"complete","run_id":"%s","steps":150,"training_code":"seed0_legacy_float32","archive_validator_exit_code":%s,"replay_validator_exit_code":%s}\n' \
    "${RUN_ID}" "${archive_validation_exit}" "${replay_validation_exit}" > "${RUN_ROOT}/summary.json"
[[ ${replay_validation_exit} -eq 0 ]]
