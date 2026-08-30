#!/usr/bin/env bash
#SBATCH --job-name=gigpo_b100_formal
#SBATCH --partition=c23g
#SBATCH --account=rwth2082
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
# Step 100->150 is expected to take about 3 hours at the validated smoke
# throughput; keep a 6-hour wall-time allowance for startup/checkpoint I/O.
#SBATCH --time=06:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
export GIGPO_FORK_RUN_NAME=gigpo_from_bace2gpu_step100_seed0
export GIGPO_FORK_INVOCATION_ID=${SLURM_JOB_ID}_step100_to150
export TARGET_STEP=150
export SAVE_FREQ=5
export MAX_CHECKPOINTS=2
export TEST_FREQ=5
export VAL_BEFORE_TRAIN=true
export VALIDATE_ON_LAST_STEP=true

bash "${SCRIPT_DIR}/run_gigpo_from_bace100_2gpu.sh"
