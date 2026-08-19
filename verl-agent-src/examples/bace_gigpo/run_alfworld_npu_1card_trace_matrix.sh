#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

exp_root=${BACE_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
seed_list=${BACE_SEEDS:-"0 1"}
train_size=${TRAIN_DATA_SIZE:-2}
epochs=${BACE_TOTAL_EPOCHS:-2}
leaf_budget=${BACE_TOTAL_LEAF_BUDGET:-4}
pilot_roots=${BACE_PILOT_ROOTS:-2}
run_prefix=${BACE_RUN_PREFIX:-bace_1card_multitask_trace_20260807}
python_bin=${BACE_PYTHON:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld/bin/python}
invalid_action_mode=${BACE_INVALID_ACTION_MODE:-strict_identity}

if [[ ! -x "${python_bin}" ]]; then
    echo "BACE validator Python is not executable: ${python_bin}" >&2
    exit 1
fi
case "${invalid_action_mode}" in
    strict_identity|valid_only_branch|single_invalid_bucket) ;;
    *)
        echo "Unknown BACE_INVALID_ACTION_MODE: ${invalid_action_mode}" >&2
        exit 1
        ;;
esac

mkdir -p "${exp_root}/logs" "${exp_root}/trace_validation"

for seed in ${seed_list}; do
    run_name="${run_prefix}_seed${seed}"
    log_path="${exp_root}/logs/${run_name}.log"
    trace_dir="${exp_root}/rollout_trajectories/${run_name}/bace_trace"
    report_path="${exp_root}/trace_validation/${run_name}.json"

    BACE_RUN_NAME="${run_name}" \
    TRAIN_DATA_SIZE="${train_size}" \
    VAL_DATA_SIZE=1 \
    BACE_TOTAL_LEAF_BUDGET="${leaf_budget}" \
    BACE_PILOT_ROOTS="${pilot_roots}" \
    BACE_BRANCH_COUNT=2 \
    bash "${SCRIPT_DIR}/run_alfworld_npu_1card_staged_smoke.sh" \
        env.seed="${seed}" \
        trainer.total_epochs="${epochs}" \
        algorithm.bace.invalid_action_mode="${invalid_action_mode}" \
        algorithm.bace.artifacts.enabled=true \
        2>&1 | tee "${log_path}"

    "${python_bin}" -m recipe.bace_gigpo.validate_trace \
        "${trace_dir}" \
        --output "${report_path}"
done
