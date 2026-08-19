#!/usr/bin/env bash
set -euo pipefail

# BACE-GiGPO BatchERV Exact main run. Runtime/memory settings are delegated to
# the validated GiGPO-derived 8-card script; only method settings are overridden.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export AESC_EXP_ROOT=${AESC_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
export AESC_RUN_NAME=${AESC_RUN_NAME:-bace_alfworld_npu_8card_150epoch_batch_erv_exact}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-64}
export GROUP_SIZE=${GROUP_SIZE:-8}
export ENV_MAX_STEPS=${ENV_MAX_STEPS:-40}

exec bash "${SCRIPT_DIR}/run_bace_alfworld_npu_8card_frontier.sh" \
    algorithm.adv_estimator=bace_gigpo \
    algorithm.bace.enabled=true \
    algorithm.bace.variant=batch_erv_exact \
    algorithm.bace.topology=dynamic \
    algorithm.bace.dynamic_root_generation=staged \
    algorithm.bace.staged_root_batching=packed \
    algorithm.bace.acquisition=batch_erv_exact \
    algorithm.bace.total_leaf_budget=8 \
    algorithm.bace.min_natural_roots=2 \
    algorithm.bace.competence_threshold=0.5 \
    algorithm.bace.history_base_alpha=1.0 \
    algorithm.bace.history_base_beta=1.0 \
    algorithm.bace.history_forgetting=0.9 \
    algorithm.bace.history_transfer_fraction=0.1 \
    algorithm.bace.history_min_strength=2.0 \
    algorithm.bace.history_max_strength=8.0 \
    algorithm.bace.local_prior_strength=2.0 \
    algorithm.bace.max_branches_per_anchor=2 \
    algorithm.bace.batch_erv_threshold=0.01 \
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12 \
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10 \
    algorithm.bace.invalid_action_mode=strict_identity \
    algorithm.bace.local_credit_mode=occurrence \
    algorithm.bace.artifacts.enabled=true \
    algorithm.bace.artifacts.include_token_arrays=true \
    algorithm.bace.artifacts.fsync=false \
    algorithm.bace.replay.compare_action_set=true \
    algorithm.bace.replay.max_origin_retries=1 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    algorithm.gamma=0.95 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    "$@"
