#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/opt/dpcvol/datasets/8165423358032568398/alfworld}
unset VLLM_ATTENTION_BACKEND LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

CONDA_ENV=${CONDA_ENV:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld}
CONDA_SH=${CONDA_SH:-/home/naie/Asend/miniconda3/etc/profile.d/conda.sh}
[[ -f "${CONDA_SH}" ]] || { echo "Conda activation script not found: ${CONDA_SH}" >&2; exit 1; }
export JAVA_HOME=${JAVA_HOME:-}
export JAVA_LD_LIBRARY_PATH=${JAVA_LD_LIBRARY_PATH:-}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

ENGINE=${ENGINE:-vllm}
filtered_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_url=*|--data_url=*) shift ;;
        --checkpoint_url|--data_url)
            shift
            [[ $# -eq 0 ]] || shift
            ;;
        *) filtered_args+=("$1"); shift ;;
    esac
done
if [[ ${#filtered_args[@]} -gt 0 && "${filtered_args[0]}" != *=* && "${filtered_args[0]}" != --* ]]; then
    ENGINE=${filtered_args[0]}
    filtered_args=("${filtered_args[@]:1}")
fi
hydra_overrides=("${filtered_args[@]}")

exp_root=${BACE_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_name=${BACE_RUN_NAME:-bace_alfworld_npu_8card_full_strict_${run_timestamp}}
data_root=${BACE_DATA_ROOT:-${exp_root}/data/${run_name}}
checkpoint_dir=${CHECKPOINT_DIR:-${exp_root}/checkpoints/${run_name}}
rollout_data_dir=${ROLLOUT_DATA_DIR:-${exp_root}/rollout_trajectories/${run_name}}
tensorboard_dir=${TENSORBOARD_DIR:-${exp_root}/tensorboard/${run_name}}
metadata_dir="${exp_root}/run_metadata/${run_name}"
validation_report="${exp_root}/trace_validation/${run_name}.json"
log_file="${exp_root}/logs/${run_name}.log"
anchor_dump_dir=${ASEC_ANCHOR_DUMP_DIR:-${exp_root}/anchor_groups/${run_name}}
export TENSORBOARD_DIR="${tensorboard_dir}"
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${exp_root}/swanlab/${run_name}}
export ASEC_ANCHOR_DUMP_DIR="${anchor_dump_dir}"
mkdir -p "${data_root}" "${checkpoint_dir}" "${rollout_data_dir}" "${tensorboard_dir}" "${metadata_dir}" "${anchor_dump_dir}" "${SWANLAB_LOG_DIR}" "$(dirname "${validation_report}")" "$(dirname "${log_file}")"
exec > >(tee -a "${log_file}") 2>&1
exec 19>> "${metadata_dir}/shell_trace.log"
export BASH_XTRACEFD=19
PS4='+ ${BASH_SOURCE}:${LINENO}: '

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-64}
group_size=${GROUP_SIZE:-8}
mode=${GIGPO_MODE:-mean_std_norm}
max_steps=${ENV_MAX_STEPS:-40}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
model_path=${MODEL_PATH:-/opt/dpcvol/datasets/8165423358032568398/model/Qwen2.5-1.5B-Instruct}
rollout_tp=${ROLLOUT_TP:-1}
logger_backends=${LOGGER_BACKENDS:-"['console','tensorboard']"}
total_epochs=${BACE_TOTAL_EPOCHS:-150}
resume_mode=${BACE_RESUME_MODE:-disable}
start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'run_name=%q\nstart_time=%q\nrepo_root=%q\ngit_head=%q\n' "${run_name}" "${start_time}" "${REPO_ROOT}" "$(git rev-parse HEAD)"
    printf 'engine=%q\nresume_mode=%q\nvisible_devices=%q\nmodel_path=%q\n' "${ENGINE}" "${resume_mode}" "${ASCEND_RT_VISIBLE_DEVICES}" "${model_path}"
    printf 'train_data_size=%q\nval_data_size=%q\ngroup_size=%q\nmax_steps=%q\nmax_response_length=%q\n' "${train_data_size}" "${val_data_size}" "${group_size}" "${max_steps}" "${max_response_length}"
    printf 'BACE topology=dynamic generation=staged acquisition=erv total_leaf_budget=8 pilot_roots=2 competence_threshold=0.35 history_forgetting=0.95 invalid_action_mode=strict_identity\n'
    env | sort
    git status --short
} > "${metadata_dir}/run_metadata.txt"
: > "${metadata_dir}/hydra_overrides.txt"
for override in "${hydra_overrides[@]}"; do printf '%q\n' "${override}" >> "${metadata_dir}/hydra_overrides.txt"; done
git diff --check > "${metadata_dir}/git_diff_check.txt" || true
npu-smi info > "${metadata_dir}/npu_smi_before.txt" 2>&1 || true

run_finalized=0
finalize_run() {
    local exit_code=$1
    if [[ ${run_finalized} -eq 1 ]]; then return; fi
    run_finalized=1
    trap - EXIT INT TERM
    set +e
    if [[ -d "${rollout_data_dir}/bace_trace" ]] && find "${rollout_data_dir}/bace_trace" -name summary.json -print -quit | grep -q .; then
        python3 -m recipe.bace_gigpo.validate_trace "${rollout_data_dir}/bace_trace" --output "${validation_report}"
        printf 'trace_validation_exit_code=%s\n' "$?" >> "${metadata_dir}/run_metadata.txt"
    else
        printf '{"ok": false, "reason": "no complete trace found"}\n' > "${validation_report}"
    fi
    npu-smi info > "${metadata_dir}/npu_smi_after.txt" 2>&1 || true
    printf 'end_time=%s\ntrain_exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" >> "${metadata_dir}/run_metadata.txt"
}
trap 'status=$?; finalize_run "${status}"; exit "${status}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 -m examples.data_preprocess.prepare \
    --mode text --local_dir "${data_root}" \
    --train_data_size "${train_data_size}" --val_data_size "${val_data_size}"

set +e
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bace_gigpo \
    algorithm.bace.enabled=true \
    algorithm.bace.topology=dynamic \
    algorithm.bace.dynamic_root_generation=staged \
    algorithm.bace.acquisition=erv \
    algorithm.bace.total_leaf_budget=8 \
    algorithm.bace.pilot_roots=2 \
    algorithm.bace.competence_threshold=0.35 \
    algorithm.bace.history_base_alpha=1.0 \
    algorithm.bace.history_base_beta=1.0 \
    algorithm.bace.history_forgetting=0.95 \
    algorithm.bace.history_transfer_fraction=0.1 \
    algorithm.bace.history_min_strength=2.0 \
    algorithm.bace.history_max_strength=8.0 \
    algorithm.bace.max_branches_per_anchor=2 \
    algorithm.bace.erv_threshold=0.0 \
    algorithm.bace.erv_temperature=0.05 \
    algorithm.bace.erv_mc_samples=512 \
    algorithm.bace.local_prior_strength=2.0 \
    algorithm.bace.local_credit_mode=occurrence \
    algorithm.bace.invalid_action_mode=strict_identity \
    algorithm.bace.artifacts.enabled=true \
    algorithm.bace.artifacts.include_token_arrays=true \
    algorithm.bace.artifacts.fsync=false \
    algorithm.bace.replay.compare_action_set=true \
    algorithm.bace.replay.max_origin_retries=1 \
    data.train_files="${data_root}/text/train.parquet" \
    data.val_files="${data_root}/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length=2048 \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=true data.truncation=error data.return_raw_chat=true \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_invalid_action_penalty=true \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=false algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 algorithm.gigpo.mode="${mode}" \
    env.env_name=alfworld/AlfredTWEnv env.seed=0 env.max_steps="${max_steps}" \
    env.rollout.n="${group_size}" env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 trainer.logger="${logger_backends}" \
    trainer.project_name=verl_agent_alfworld trainer.experiment_name="${run_name}" \
    trainer.device=npu trainer.n_gpus_per_node=8 trainer.nnodes=1 \
    trainer.save_freq=30 trainer.default_local_dir="${checkpoint_dir}" \
    trainer.rollout_data_dir="${rollout_data_dir}" trainer.test_freq=10 \
    trainer.total_epochs="${total_epochs}" trainer.val_before_train=false \
    trainer.resume_mode="${resume_mode}" "${hydra_overrides[@]}"
train_status=$?
set -e
exit "${train_status}"
