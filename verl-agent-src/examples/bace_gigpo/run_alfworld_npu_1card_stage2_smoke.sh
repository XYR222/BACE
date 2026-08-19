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
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
unset VLLM_ATTENTION_BACKEND LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

CONDA_ENV=${CONDA_ENV:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld}
source /home/naie/Asend/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

exp_root=${BACE_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
run_name=${BACE_RUN_NAME:-bace_stage2_qwen2.5_1.5b_npu_1card_smoke}
mkdir -p "${exp_root}/checkpoints/${run_name}" "${exp_root}/rollout_trajectories/${run_name}"

train_data_size=${TRAIN_DATA_SIZE:-2}
val_data_size=${VAL_DATA_SIZE:-4}
root_count=${BACE_ROOT_COUNT:-6}
branch_count=${BACE_BRANCH_COUNT:-1}
topology=${BACE_TOPOLOGY:-fixed}
dynamic_root_generation=${BACE_DYNAMIC_ROOT_GENERATION:-preallocated}
total_leaf_budget=${BACE_TOTAL_LEAF_BUDGET:-8}
pilot_roots=${BACE_PILOT_ROOTS:-2}
if [[ "${topology}" == "dynamic" ]]; then
    root_count=${total_leaf_budget}
fi
data_root=${BACE_DATA_ROOT:-${REPO_ROOT}/data/verl-agent}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${data_root}" \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bace_gigpo \
    algorithm.bace.enabled=true \
    algorithm.bace.topology="${topology}" \
    algorithm.bace.dynamic_root_generation="${dynamic_root_generation}" \
    algorithm.bace.fixed_root_count="${root_count}" \
    algorithm.bace.fixed_branch_count="${branch_count}" \
    algorithm.bace.total_leaf_budget="${total_leaf_budget}" \
    algorithm.bace.pilot_roots="${pilot_roots}" \
    data.train_files="${data_root}/text/train.parquet" \
    data.val_files="${data_root}/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=/opt/dpcvol/datasets/8165423358032568398/model/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.use_kl_in_reward=false \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=8 \
    env.rollout.n="${root_count}" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name="${run_name}" \
    trainer.device=npu \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.default_local_dir="${exp_root}/checkpoints/${run_name}" \
    trainer.rollout_data_dir="${exp_root}/rollout_trajectories/${run_name}" \
    "$@"
