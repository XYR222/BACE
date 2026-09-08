#!/usr/bin/env bash
# One-step H100 smoke test for the upstream GiGPO ALFWorld entry point.
# It intentionally invokes examples/gigpo_trainer/run_alfworld.sh directly;
# no BACE collector, trace validator, checkpoint gate, or follow-up submitter
# is involved.
#SBATCH --job-name=gigpo-alf-smoke
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-smoke/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-smoke/slurm/%x-%j.err

set -euo pipefail

module purge
# ALFWorld/TextWorld's Fast Downward extension requires GLIBCXX_3.4.32.
# Keep this aligned with the previously successful AFH GiGPO jobs.
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent

# These ROCm variables are inherited on some cluster shells.  veRL rejects
# them when Slurm has correctly set CUDA_VISIBLE_DEVICES for an H100 job.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export TENSORBOARD_DIR=/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-smoke/tensorboard/${SLURM_JOB_ID}
mkdir -p "${TENSORBOARD_DIR}" /hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-smoke/slurm

# The command below is the original GiGPO runner.  Overrides only bound the
# smoke workload to one optimizer step; they do not enable any BACE option.
bash examples/gigpo_trainer/run_alfworld.sh vllm \
    data.train_files=/hpcwork/xsz96350/fu_project/data/text/train.parquet \
    data.val_files=/hpcwork/xsz96350/fu_project/data/text/test.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_response_length=64 \
    actor_rollout_ref.model.path=/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    env.max_steps=2 \
    env.rollout.n=2 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='gigpo_alfworld_upstream_smoke' \
    trainer.experiment_name="h100_2gpu_${SLURM_JOB_ID}" \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=false \
    trainer.validate_on_last_step=false
