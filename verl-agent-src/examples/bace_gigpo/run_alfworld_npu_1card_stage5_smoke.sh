#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BACE_TOPOLOGY=dynamic
export BACE_TOTAL_LEAF_BUDGET=${BACE_TOTAL_LEAF_BUDGET:-6}
export BACE_PILOT_ROOTS=${BACE_PILOT_ROOTS:-2}
export BACE_BRANCH_COUNT=${BACE_BRANCH_COUNT:-2}
export BACE_RUN_NAME=${BACE_RUN_NAME:-bace_stage5_dynamic_npu_1card_smoke}

exec bash "${SCRIPT_DIR}/run_alfworld_npu_1card_stage2_smoke.sh" \
    algorithm.bace.competence_threshold=0.0 \
    "$@"
