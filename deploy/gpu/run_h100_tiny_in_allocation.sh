#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
BUNDLE_ROOT=${WORKSPACE_ROOT}/work-BACE
REPO_ROOT=${BUNDLE_ROOT}/verl-agent-src
ENV_PREFIX=${WORKSPACE_ROOT}/verl-agent

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

export MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-3B-Instruct
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
export TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
export VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
export BACE_EXP_ROOT=${BUNDLE_ROOT}/experiments/h100-tiny-smoke
export PYTHONUNBUFFERED=1

mkdir -p "${BACE_EXP_ROOT}/allocation_preflight"
preflight_report=${BACE_EXP_ROOT}/allocation_preflight/preflight_$(date -u +%Y%m%dT%H%M%SZ).json

echo "node=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${REPO_ROOT}"
python "${BUNDLE_ROOT}/deploy/gpu/check_gpu_environment.py" \
    --expected-gpus 1 \
    --model-path "${MODEL_PATH}" \
    --alfworld-data "${ALFWORLD_DATA}" \
    --output "${preflight_report}"

DRY_RUN=1 bash examples/gigpo_trainer/run_bace_alfworld_h100_tiny_smoke.sh
bash examples/gigpo_trainer/run_bace_alfworld_h100_tiny_smoke.sh
