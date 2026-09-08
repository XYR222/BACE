#!/usr/bin/env bash
# Full upstream GiGPO ALFWorld run: seed 0, 150 updates, no checkpoints.
# It calls the original examples/gigpo_trainer/run_alfworld.sh directly.
# The only Hydra overrides below bind upstream defaults to local assets and
# make the requested seed/no-checkpoint behavior explicit.
#SBATCH --job-name=gigpo-alf-s0
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=12:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/%x-%j.err

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export WANDB_DIR=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/wandb/${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}" /hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm

bash examples/gigpo_trainer/run_alfworld.sh vllm \
    data.train_files=/hpcwork/xsz96350/fu_project/data/text/train.parquet \
    data.val_files=/hpcwork/xsz96350/fu_project/data/text/test.parquet \
    actor_rollout_ref.model.path=/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct \
    env.seed=0 \
    trainer.project_name=gigpo_alfworld_upstream \
    trainer.experiment_name=gigpo_qwen2.5_1.5b_seed0 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable
