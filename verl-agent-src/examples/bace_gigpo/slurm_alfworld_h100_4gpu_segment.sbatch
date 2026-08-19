#!/usr/bin/env bash
#SBATCH --job-name=bace15b_main
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=23:30:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src/examples/bace_gigpo
source "${SCRIPT_DIR}/h100_job_env.sh"

: "${SEGMENT_END_STEP:?SEGMENT_END_STEP must be exported by the chain submitter}"
: "${ACTOR_MICRO_BATCH:?ACTOR_MICRO_BATCH must be exported by the chain submitter}"
: "${MAX_CHECKPOINTS:?MAX_CHECKPOINTS must be exported by the chain submitter}"

export BACE_EXP_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
export BACE_RUN_NAME=bace_alfworld_qwen2_5_1_5b_exact_main_seed0
export BACE_INVOCATION_ID=${SLURM_JOB_ID}_through_step_${SEGMENT_END_STEP}
export BACE_RESUME_MODE=auto
export SAVE_FREQ=5
export TEST_FREQ=5
export LOGPROB_MICRO_BATCH=${ACTOR_MICRO_BATCH}
unset CHECKPOINT_DIR BACE_ARTIFACT_DIR ROLLOUT_DATA_DIR TENSORBOARD_DIR
unset RUN_METADATA_DIR TRACE_VALIDATION_REPORT LOG_FILE

python "${SCRIPT_DIR}/check_formal_storage.py" "${BACE_EXP_ROOT}" \
    --expected-prefix /hpcwork/rwth2089/xsz96350 \
    --minimum-free-gib 64 \
    --output "${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_storage_preflight.json"

bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh"

checkpoint_root=${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}
[[ "$(<"${checkpoint_root}/latest_checkpointed_iteration.txt")" == "${SEGMENT_END_STEP}" ]]
python -m recipe.bace_gigpo.validate_trace \
    "${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}" \
    --output "${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}/${BACE_INVOCATION_ID}_final.json"
