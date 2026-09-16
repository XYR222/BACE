#!/usr/bin/env bash
# Full upstream GiGPO ALFWorld run: 150 updates, no checkpoints.
# RUN_SEED defaults to 0 and can be overridden at submission time, e.g.
#   sbatch --export=ALL,RUN_SEED=1 --job-name=gigpo-alf-s1 <this-script>
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
#SBATCH --output=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/%x-%j.out
#SBATCH --error=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/%x-%j.err

set -euo pipefail

RUN_SEED="${RUN_SEED:-0}"
if [[ ! "${RUN_SEED}" =~ ^[0-9]+$ ]]; then
    echo "RUN_SEED must be a non-negative integer, got: ${RUN_SEED}" >&2
    exit 2
fi
STEP_ADVANTAGE_W="${GIGPO_STEP_ADVANTAGE_W:-1.0}"
if [[ ! "${STEP_ADVANTAGE_W}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "GIGPO_STEP_ADVANTAGE_W must be a non-negative number, got: ${STEP_ADVANTAGE_W}" >&2
    exit 2
fi
STEP_WEIGHT_SLUG=${STEP_ADVANTAGE_W/./p}

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
unset WANDB_MODE WANDB_DIR
export TENSORBOARD_DIR=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/tensorboard/${SLURM_JOB_ID}
mkdir -p "${TENSORBOARD_DIR}" /hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm

bash examples/gigpo_trainer/run_alfworld.sh vllm \
    data.train_files=/hpcwork/xsz96350/fu_project/data/text/train.parquet \
    data.val_files=/hpcwork/xsz96350/fu_project/data/text/test.parquet \
    actor_rollout_ref.model.path=/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct \
    env.seed="${RUN_SEED}" \
    algorithm.gigpo.step_advantage_w="${STEP_ADVANTAGE_W}" \
    trainer.project_name=gigpo_alfworld_upstream \
    trainer.experiment_name="gigpo_qwen2.5_1.5b_seed${RUN_SEED}_w${STEP_WEIGHT_SLUG}" \
    "trainer.logger=['console','tensorboard']" \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable
