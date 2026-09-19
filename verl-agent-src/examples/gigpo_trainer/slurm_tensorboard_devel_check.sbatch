#!/usr/bin/env bash
# CPU-only preflight for the TensorBoard logger used by the full GiGPO wrapper.
# This does not initialize Ray, vLLM, ALFWorld, or any GPU worker.
#SBATCH --job-name=gigpo-tb-check
#SBATCH --partition=devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/preflight/%x-%j.out
#SBATCH --error=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/preflight/%x-%j.err

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent

unset WANDB_MODE WANDB_DIR VLLM_ATTENTION_BACKEND ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export TENSORBOARD_DIR=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/tensorboard/preflight-${SLURM_JOB_ID}
mkdir -p "${TENSORBOARD_DIR}" /hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/preflight

# Exercise the exact veRL Tracking implementation rather than only importing
# tensorboard.  A real event file and scalar must be flushed successfully.
python3 - <<'PY'
import os
import sys
from pathlib import Path

from verl.utils.tracking import Tracking

tracking = Tracking(
    project_name="gigpo_alfworld_upstream",
    experiment_name="tensorboard_devel_preflight",
    default_backend=["console", "tensorboard"],
)
tracking.log({"preflight/tensorboard_write": 1.0}, step=1)
tracking.logger["tensorboard"].writer.flush()
tracking.logger["tensorboard"].finish()

event_files = list(Path(os.environ["TENSORBOARD_DIR"]).glob("events.out.tfevents.*"))
assert event_files, "TensorBoard did not create an event file"
assert "wandb" not in tracking.logger
assert "wandb" not in sys.modules
print(f"tensorboard_preflight=ok event_file={event_files[0]}")
PY

# Resolve the exact logger override accepted by the full GiGPO invocation.
# Hydra --cfg job exits before Ray/GPU initialization.
python3 -m verl.trainer.main_ppo \
    "trainer.logger=['console','tensorboard']" \
    trainer.project_name=gigpo_alfworld_upstream \
    trainer.experiment_name=tensorboard_devel_preflight \
    --cfg job --resolve \
    > /hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-upstream-full/preflight/resolved_config-${SLURM_JOB_ID}.yaml

printf 'tensorboard_and_hydra_preflight=ok\n'
