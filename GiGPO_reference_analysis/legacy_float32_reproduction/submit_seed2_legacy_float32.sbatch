#!/usr/bin/env bash
#SBATCH --job-name=gigpo_ref_s2_f32
#SBATCH --partition=c23g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=12:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/GiGPO_reference_analysis/legacy_float32_reproduction
#SBATCH --output=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-reference/slurm/%x-%j.out
#SBATCH --error=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-reference/slurm/%x-%j.err

set -euo pipefail
export REF_SEED=2
export REF_RUN_ID=gigpo_ref_v2_seed2_legacy_float32
export REF_RUN_ROOT=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-reference/runs/seed_2_legacy_float32_v1
exec bash ./run_legacy_float32_reference.sh
