#!/usr/bin/env bash
#SBATCH --job-name=bace15b_1g_debug
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_INVOCATION_ID=${SLURM_JOB_ID}_1gpu_debug

bash "${SCRIPT_DIR}/run_alfworld_h100_1gpu_debug.sh"
python -m recipe.bace_gigpo.validate_trace \
    "${BACE_EXP_ROOT}/bace_artifacts/bace_alfworld_qwen2_5_1_5b_1gpu_debug" \
    --output "${BACE_EXP_ROOT}/trace_validation/bace_alfworld_qwen2_5_1_5b_1gpu_debug/${BACE_INVOCATION_ID}_final.json"
grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" \
    "${BACE_EXP_ROOT}/logs/bace_alfworld_qwen2_5_1_5b_1gpu_debug/${BACE_INVOCATION_ID}.log"
