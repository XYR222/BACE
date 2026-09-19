#!/usr/bin/env bash
# Disposable one-GPU branch/Replay gate for the correction=1 C0--C8 methods.
# This is deliberately a debug configuration, not a formal-training launcher.
# Submit with BACE_METHOD in {c0,c1,c2,c3,c05,c4,c7,c8}.
#SBATCH --job-name=bace-smoke1
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

: "${BACE_METHOD:?set BACE_METHOD to c0, c1, c2, c3, c05, c4, c7, or c8}"
REPO=${BACE_REPO_ROOT:-/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src}
[[ -f "${REPO}/examples/bace_gigpo/h100_job_env.sh" ]] || {
    echo "BACE_REPO_ROOT is not a verl-agent-src checkout: ${REPO}" >&2
    exit 2
}

source "${REPO}/examples/bace_gigpo/h100_job_env.sh"
unset VLLM_ATTENTION_BACKEND
export GPU_COUNT=1 ROLLOUT_TP=1 BACE_SEED=0
export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_RUN_NAME=${BACE_RUN_NAME:-preformal_1gpu_smoke_${BACE_METHOD}_s0_${SLURM_JOB_ID}}
export BACE_INVOCATION_ID=${BACE_INVOCATION_ID:-${SLURM_JOB_ID}_${BACE_METHOD}}

case "${BACE_METHOD}" in
    c0)  credit=(algorithm.bace.credit_mode=current algorithm.bace.tree_credit_mode=current) ;;
    c1)  credit=(algorithm.bace.credit_mode=null algorithm.bace.tree_credit_mode=o1_local) ;;
    c2)  credit=(algorithm.bace.credit_mode=null algorithm.bace.tree_credit_mode=o1_tree_macro) ;;
    c3)  credit=(algorithm.bace.credit_mode=null algorithm.bace.tree_credit_mode=o1_full_tree) ;;
    c05) credit=(algorithm.bace.credit_mode=c0_5_origin_family_local_mean algorithm.bace.tree_credit_mode=current) ;;
    c4)  credit=(algorithm.bace.credit_mode=c4_macro_strict_ancestor algorithm.bace.tree_credit_mode=current) ;;
    c7)  credit=(algorithm.bace.credit_mode=c7_flat_leaf_gigpo algorithm.bace.tree_credit_mode=current) ;;
    c8)  credit=(algorithm.bace.credit_mode=c8_macro_local_strict_ancestor algorithm.bace.tree_credit_mode=current) ;;
    *) echo "unknown BACE_METHOD: ${BACE_METHOD}" >&2; exit 2 ;;
esac

# This is deliberately a real Hydra override, not an environment hint.  The
# Beta(1.2, 0.8) cold-start prior makes first-step branch readiness likely.
bash "${REPO}/examples/bace_gigpo/run_alfworld_h100_1gpu_debug.sh" \
    env.seed=0 \
    algorithm.bace.variant=batch_erv_exact \
    algorithm.bace.acquisition=batch_erv_exact \
    algorithm.bace.topology=dynamic \
    algorithm.bace.dynamic_root_generation=staged \
    algorithm.bace.staged_root_batching=packed \
    algorithm.bace.branch_execution_mode=selected_worker \
    algorithm.bace.branch_pool_mode=main_reuse \
    algorithm.bace.root_active_executor=true \
    algorithm.bace.total_leaf_budget=8 \
    algorithm.bace.min_natural_roots=2 \
    algorithm.bace.capacity_correction_batch_size=1 \
    algorithm.bace.max_branches_per_anchor=2 \
    algorithm.bace.competence_threshold=0.5 \
    algorithm.bace.batch_erv_threshold=0.005 \
    algorithm.bace.tree_ppo_padding_mode=copy_trainable \
    "${credit[@]}" \
    algorithm.bace.history_initial_mean=0.60

[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

python -m recipe.bace_gigpo.validate_trace \
    "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
    --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${SLURM_JOB_ID}_final.json"
python "${REPO}/examples/bace_gigpo/validate_selected_worker_metrics.py" \
    "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
    --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${SLURM_JOB_ID}_selected_worker.json"
