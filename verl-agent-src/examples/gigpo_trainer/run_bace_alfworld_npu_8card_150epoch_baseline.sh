#!/usr/bin/env bash
set -euo pipefail

# BACE-GiGPO main experiment: 8-card NPU, 150 epochs.
#
# This is the measured baseline orchestration path: it reuses the validated
# frontier script and explicitly disables the optional frontier coalescing
# path.  All rollout, replay, root, branch, and token artifacts are written
# below AESC_EXP_ROOT by the delegated script.
#
# Method parameters are frozen from:
# BACE_GiGPO_实验参数选择与推荐配置_最终版.md
#
# The non-BACE runtime settings remain those of the successful GiGPO 8-card
# script: TP=1, ref/rollout micro-batch=8, max_steps=40, and the NPU memory
# settings defined in run_bace_alfworld_npu_8card_frontier.sh. Actor PPO
# micro-batch is reduced to 4 because the mb=8 run exhausted NPU memory in
# aclnnFlashAttentionUnpaddingScoreGrad; ppo_mini_batch_size remains 128.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export AESC_EXP_ROOT=${AESC_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
export AESC_RUN_NAME=${AESC_RUN_NAME:-bace_alfworld_npu_8card_150epoch_baseline_fix3_mb4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-64}
export GROUP_SIZE=${GROUP_SIZE:-8}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}

# Keep the baseline's dispatch semantics.  max_batch_size/min_batch_size are
# still passed for a complete, auditable configuration, but the disabled
# switch makes them inert.
export BACE_COALESCING_ENABLED=false

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_frontier.sh" \
    algorithm.adv_estimator=bace_gigpo \
    algorithm.bace.enabled=true \
    algorithm.bace.topology=dynamic \
    algorithm.bace.dynamic_root_generation=staged \
    algorithm.bace.staged_root_batching=frontier \
    algorithm.bace.frontier_batch_coalescing.enabled=false \
    algorithm.bace.frontier_batch_coalescing.max_batch_size=0 \
    algorithm.bace.frontier_batch_coalescing.min_batch_size=1 \
    algorithm.bace.acquisition=erv \
    algorithm.bace.total_leaf_budget=8 \
    algorithm.bace.pilot_roots=2 \
    algorithm.bace.max_branches_per_anchor=2 \
    algorithm.bace.competence_threshold=0.5 \
    algorithm.bace.history_base_alpha=1.0 \
    algorithm.bace.history_base_beta=1.0 \
    algorithm.bace.history_forgetting=0.8 \
    algorithm.bace.history_transfer_fraction=0.1 \
    algorithm.bace.history_min_strength=2.0 \
    algorithm.bace.history_max_strength=8.0 \
    algorithm.bace.local_prior_strength=2.0 \
    algorithm.bace.erv_mc_samples=512 \
    algorithm.bace.erv_threshold=0.01 \
    algorithm.bace.erv_temperature=0.02 \
    algorithm.bace.local_credit_mode=occurrence \
    algorithm.bace.invalid_action_mode=strict_identity \
    algorithm.bace.artifacts.enabled=true \
    algorithm.bace.artifacts.include_token_arrays=true \
    algorithm.bace.artifacts.fsync=false \
    algorithm.bace.replay.compare_action_set=true \
    algorithm.bace.replay.max_origin_retries=1 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    algorithm.gamma=0.95 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    "$@"
