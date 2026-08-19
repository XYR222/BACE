#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
BUNDLED_CONDA_ENV=${BUNDLED_CONDA_ENV:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld}

CONDA_SH=${CONDA_SH:-/home/naie/Asend/miniconda3/etc/profile.d/conda.sh}
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}"

if [[ -z "${CONDA_ENV:-}" ]]; then
    conda_envs=$(conda env list)
    if [[ -x "${BUNDLED_CONDA_ENV}/bin/python" ]]; then
        CONDA_ENV="${BUNDLED_CONDA_ENV}"
    elif grep -Eq "(^|[[:space:]])verl-agent-alfworld([[:space:]]|$)" <<<"${conda_envs}"; then
        CONDA_ENV=verl-agent-alfworld
    elif grep -Eq "(^|[[:space:]])verl-agent([[:space:]]|$)" <<<"${conda_envs}"; then
        CONDA_ENV=verl-agent
    else
        echo "No suitable conda environment found. Set CONDA_ENV explicitly or provide ${BUNDLED_CONDA_ENV}." >&2
        conda env list >&2
        exit 1
    fi
fi
conda activate "${CONDA_ENV}"

ENGINE=vllm
if [[ $# -gt 0 && "$1" != *"="* ]]; then
    ENGINE=$1
    shift
fi

hydra_overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_url=*|--data_url=*)
            shift
            ;;
        --checkpoint_url|--data_url)
            shift
            if [[ $# -gt 0 ]]; then
                shift
            fi
            ;;
        *)
            hydra_overrides+=("$1")
            shift
            ;;
    esac
done

# 8-card Ascend 910B/NPU 150-epoch run for GiGPO + ALFWorld.
# This uses TP=1, micro-batch 8, max_steps=40, and reduced validation cost.
unset VLLM_ATTENTION_BACKEND
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

exp_root=${AESC_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
run_name=${AESC_RUN_NAME:-gigpo_qwen2.5_1.5b_npu_8card_150epoch_tp1_mb8_s40_lowval}
checkpoint_dir=${CHECKPOINT_DIR:-${exp_root}/checkpoints/${run_name}}
rollout_data_dir=${ROLLOUT_DATA_DIR:-${exp_root}/rollout_trajectories/${run_name}}
anchor_dump_dir=${ASEC_ANCHOR_DUMP_DIR:-${exp_root}/anchor_groups/${run_name}}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${exp_root}/tensorboard/${run_name}}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${exp_root}/swanlab/${run_name}}
export ASEC_ANCHOR_DUMP_DIR="${anchor_dump_dir}"
mkdir -p \
    "${checkpoint_dir}" \
    "${rollout_data_dir}" \
    "${anchor_dump_dir}" \
    "${TENSORBOARD_DIR}" \
    "${SWANLAB_LOG_DIR}"

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

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files="${HOME}/data/verl-agent/text/train.parquet" \
    data.val_files="${HOME}/data/verl-agent/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length=2048 \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="${mode}" \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps="${max_steps}" \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 \
    trainer.logger="${logger_backends}" \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name="${run_name}" \
    trainer.device=npu \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    "${hydra_overrides[@]}"
