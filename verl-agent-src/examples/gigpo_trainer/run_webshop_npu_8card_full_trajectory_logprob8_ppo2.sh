#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# PPO=4 still OOMed in FlashAttentionScoreGrad during actor backward. Keep the
# proven log-prob setting and halve only the PPO update micro-batch again.
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_fulltraj_spa8_logprob8_ppo2_${run_timestamp}}

printf '%s\n' \
    'WebShop eight-card actor-update memory diagnostic:' \
    '  PPO micro-batch per GPU: 2' \
    '  rollout/ref log-prob micro-batch per GPU: 8' \
    '  vLLM GPU memory utilization: 0.6'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_full_trajectory_logprob8.sh" "$@"
