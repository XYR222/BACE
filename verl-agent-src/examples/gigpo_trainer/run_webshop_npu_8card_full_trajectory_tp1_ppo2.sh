#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Keep the proven full-trajectory memory settings and change only rollout TP.
# TP=1 creates eight rollout replicas on eight NPUs, compared with four at TP=2.
export N_GPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ROLLOUT_TP=1
export PPO_MICRO_BATCH_SIZE_PER_GPU=2
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
export GIGPO_RUN_NAME=${GIGPO_RUN_NAME:-gigpo_webshop_npu_8card_fulltraj_tp1_spa8_logprob8_ppo2_${run_timestamp}}

printf '%s\n' \
    'WebShop eight-card TP=1 full-trajectory comparison:' \
    '  rollout tensor parallel size: 1' \
    '  rollout replicas: 8' \
    '  PPO micro-batch per GPU: 2' \
    '  rollout/ref log-prob micro-batch per GPU: 8' \
    '  max environment steps: 15' \
    '  validation/checkpoint frequency: every step'

exec bash "${SCRIPT_DIR}/run_webshop_npu_8card_full_trajectory_logprob8_ppo2.sh" "$@"
