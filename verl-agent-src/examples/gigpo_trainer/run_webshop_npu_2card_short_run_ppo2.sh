#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Two-card 8/8/8/0.6 reached actor backward and OOMed. The eight-card PPO=4
# result also OOMed, so test PPO=2 directly while retaining proven log-prob=8.
export PPO_MICRO_BATCH_SIZE_PER_GPU=4
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_2card_short_ppo2_logprob8_${run_timestamp}}

printf '%s\n' \
    'WebShop two-card actor-update memory diagnostic:' \
    '  PPO micro-batch per GPU: 4' \
    '  rollout/ref log-prob micro-batch per GPU: 4' \
    '  vLLM GPU memory utilization: 0.6'

exec bash "${SCRIPT_DIR}/run_webshop_npu_2card_short_run.sh" "$@"
