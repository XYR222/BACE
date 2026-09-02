#!/usr/bin/env bash
#SBATCH --job-name=bace_c123_gate
#SBATCH --partition=devel
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

: "${SMOKE_JOB_ID:?SMOKE_JOB_ID is required}"
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
mkdir -p "${EXP_ROOT}/run_metadata"

module purge
module load GCCcore/13.3.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent
export PYTHONNOUSERSITE=1

for mode in o1_local o1_tree_macro o1_full_tree; do
    run_name="tree_credit_${mode}_smoke_seed0_${SMOKE_JOB_ID}"
    artifact_dir="${EXP_ROOT}/bace_artifacts/${run_name}"
    report="${EXP_ROOT}/trace_validation/${run_name}/${SMOKE_JOB_ID}_${mode}_gate_recheck.json"
    python "${SCRIPT_DIR}/validate_tree_credit_smoke.py" \
        "${artifact_dir}" --mode "${mode}" --output "${report}"
done

bash "${SCRIPT_DIR}/submit_tree_credit_c123_2gpu_seed0.sh" | \
    tee "${EXP_ROOT}/run_metadata/tree_credit_c123_gate_${SMOKE_JOB_ID}.txt"
