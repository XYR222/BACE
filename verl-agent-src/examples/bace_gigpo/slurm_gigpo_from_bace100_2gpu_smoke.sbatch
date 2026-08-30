#!/usr/bin/env bash
#SBATCH --job-name=gigpo_b100_smoke
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export GIGPO_FORK_RUN_NAME=gigpo_from_bace2gpu_step100_smoke_${SLURM_JOB_ID}
export GIGPO_FORK_INVOCATION_ID=${SLURM_JOB_ID}_restore_step101
export TARGET_STEP=101
export SAVE_FREQ=1
export MAX_CHECKPOINTS=1
export TEST_FREQ=-1
export VAL_BEFORE_TRAIN=false
export VALIDATE_ON_LAST_STEP=false

bash "${SCRIPT_DIR}/run_gigpo_from_bace100_2gpu.sh"

log_file=${EXP_ROOT}/logs/${GIGPO_FORK_RUN_NAME}/${GIGPO_FORK_INVOCATION_ID}.log
grep -F "Load from checkpoint folder:" "${log_file}" | grep -F "global_step_100"
grep -F "Setting global step to 100" "${log_file}"
grep -Eq "training/global_step['\"]?:[[:space:]]*101|training/global_step[^0-9]+101" "${log_file}"
! grep -Eq "bace/(natural_roots|executed_branches|batch_erv_exact)" "${log_file}"

gate_dir=${EXP_ROOT}/run_metadata/gigpo_from_bace2gpu_step100_gate
mkdir -p "${gate_dir}"
printf 'smoke_job_id=%s\nsmoke_run=%s\nstatus=passed\n' \
    "${SLURM_JOB_ID}" "${GIGPO_FORK_RUN_NAME}" > "${gate_dir}/${SLURM_JOB_ID}.txt"

if [[ "${SUBMIT_FORMAL:-1}" == 1 ]]; then
    formal_job=$(sbatch --parsable "${SCRIPT_DIR}/slurm_gigpo_from_bace100_2gpu_formal.sbatch")
    printf 'formal_job_id=%s\n' "${formal_job}" | tee -a "${gate_dir}/${SLURM_JOB_ID}.txt"
fi
