#!/usr/bin/env bash

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
BUNDLE_ROOT=${WORKSPACE_ROOT}/work-BACE
REPO_ROOT=${BUNDLE_ROOT}/verl-agent-src
ENV_PREFIX=${WORKSPACE_ROOT}/verl-agent

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# ALFWorld uses one Ray actor per environment.  The formal configuration owns
# 272 actors (train, validation, and Replay pools), so allowing NumPy/PyTorch
# to create a full BLAS/OpenMP pool in every actor exhausts the Slurm task's
# PID/thread cgroup after a few rollout waves.  Keep native libraries
# single-threaded; environment-level Ray parallelism already supplies all
# required CPU concurrency.
export BACE_CPU_THREADS=${BACE_CPU_THREADS:-1}
export OMP_NUM_THREADS=${BACE_CPU_THREADS}
export OMP_THREAD_LIMIT=${BACE_CPU_THREADS}
export MKL_NUM_THREADS=${BACE_CPU_THREADS}
export OPENBLAS_NUM_THREADS=${BACE_CPU_THREADS}
export NUMEXPR_NUM_THREADS=${BACE_CPU_THREADS}
export NUMEXPR_MAX_THREADS=${BACE_CPU_THREADS}
export VECLIB_MAXIMUM_THREADS=${BACE_CPU_THREADS}
export RAYON_NUM_THREADS=${BACE_CPU_THREADS}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
# Ray otherwise detects every physical CPU on the H100 node and eagerly starts
# workers outside the 64 CPUs granted to this Slurm task.
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}
export MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
export VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

cd "${REPO_ROOT}"
