#!/usr/bin/env bash
#SBATCH --job-name=bace_h100_tiny_0817
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/h100-tiny-smoke/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/h100-tiny-smoke/slurm/%x-%j.err

set -euo pipefail

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gres=gpu:1 \
    /hpcwork/xsz96350/fu_project/work-BACE/deploy/gpu/run_h100_tiny_in_allocation.sh
