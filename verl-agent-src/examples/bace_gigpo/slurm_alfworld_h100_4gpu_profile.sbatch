#!/usr/bin/env bash
#SBATCH --job-name=bace15b_4g_profile
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

FORMAL_ROOT=/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact
PROFILE_ROOT=${FORMAL_ROOT}/recovery_profile
mkdir -p "${PROFILE_ROOT}/profile_reports" "${FORMAL_ROOT}/chain"

is_cuda_oom() {
    grep -Eiq \
        'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError: CUDA|CUDA error: out of memory' \
        "$@" 2>/dev/null
}

remove_failed_checkpoint() {
    local checkpoint_path=$1
    python - "${PROFILE_ROOT}/checkpoints" "${checkpoint_path}" <<'PY'
import shutil
import sys
from pathlib import Path

parent = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target.parent != parent or not target.name.startswith("profile_mb"):
    raise SystemExit(f"refusing unsafe profile checkpoint cleanup: {target}")
if target.exists():
    shutil.rmtree(target)
PY
}

for micro_batch in 32 16 8; do
    attempt_name=profile_mb${micro_batch}_job${SLURM_JOB_ID}
    checkpoint_dir=${PROFILE_ROOT}/checkpoints/${attempt_name}
    artifact_dir=${PROFILE_ROOT}/bace_artifacts/${attempt_name}
    phase1_id=${SLURM_JOB_ID}_${attempt_name}_step1
    phase2_id=${SLURM_JOB_ID}_${attempt_name}_step2
    phase1_log=${PROFILE_ROOT}/logs/${attempt_name}/${phase1_id}.log
    phase2_log=${PROFILE_ROOT}/logs/${attempt_name}/${phase2_id}.log

    phase1_start=$(date +%s)
    set +e
    BACE_EXP_ROOT=${PROFILE_ROOT} BACE_RUN_NAME=${attempt_name} \
      BACE_INVOCATION_ID=${phase1_id} CHECKPOINT_DIR=${checkpoint_dir} \
      BACE_ARTIFACT_DIR=${artifact_dir} ACTOR_MICRO_BATCH=${micro_batch} \
      LOGPROB_MICRO_BATCH=${micro_batch} SAVE_FREQ=1 TEST_FREQ=5 \
      BACE_RESUME_MODE=disable SEGMENT_END_STEP=1 MAX_CHECKPOINTS=2 \
      bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh"
    phase1_status=$?
    set -e
    phase1_seconds=$(( $(date +%s) - phase1_start ))
    if [[ ${phase1_status} -ne 0 ]]; then
        if is_cuda_oom "${phase1_log}"; then
            remove_failed_checkpoint "${checkpoint_dir}"
            continue
        fi
        exit "${phase1_status}"
    fi

    phase2_start=$(date +%s)
    set +e
    BACE_EXP_ROOT=${PROFILE_ROOT} BACE_RUN_NAME=${attempt_name} \
      BACE_INVOCATION_ID=${phase2_id} CHECKPOINT_DIR=${checkpoint_dir} \
      BACE_ARTIFACT_DIR=${artifact_dir} ACTOR_MICRO_BATCH=${micro_batch} \
      LOGPROB_MICRO_BATCH=${micro_batch} SAVE_FREQ=1 TEST_FREQ=5 \
      BACE_RESUME_MODE=auto SEGMENT_END_STEP=2 MAX_CHECKPOINTS=2 \
      bash "${SCRIPT_DIR}/run_alfworld_h100_4gpu.sh"
    phase2_status=$?
    set -e
    phase2_seconds=$(( $(date +%s) - phase2_start ))
    if [[ ${phase2_status} -ne 0 ]]; then
        if is_cuda_oom "${phase2_log}"; then
            remove_failed_checkpoint "${checkpoint_dir}"
            continue
        fi
        exit "${phase2_status}"
    fi

    continuity_report=${PROFILE_ROOT}/profile_reports/${attempt_name}_resume_continuity.json
    python "${SCRIPT_DIR}/validate_resume_continuity.py" \
        "${checkpoint_dir}" "${artifact_dir}" \
        --output "${continuity_report}"
    python -m recipe.bace_gigpo.validate_trace "${artifact_dir}" \
        --output "${PROFILE_ROOT}/profile_reports/${attempt_name}_trace_validation.json"
    grep -Eq "training/global_step['\"]?:[[:space:]]*2|training/global_step[^0-9]+2" "${phase2_log}"

    keep_report=${PROFILE_ROOT}/profile_reports/${attempt_name}_checkpoint_capacity.json
    python "${SCRIPT_DIR}/prune_checkpoints.py" "${checkpoint_dir}" \
        --recommend --reserve-gib 10 --output "${keep_report}"
    max_checkpoints=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_keep"])' "${keep_report}")
    python "${SCRIPT_DIR}/prune_checkpoints.py" "${checkpoint_dir}" \
        --keep "${max_checkpoints}" \
        --output "${PROFILE_ROOT}/profile_reports/${attempt_name}_checkpoint_prune.json"

    slower_seconds=${phase1_seconds}
    if (( phase2_seconds > slower_seconds )); then slower_seconds=${phase2_seconds}; fi
    (( slower_seconds > 0 )) || slower_seconds=1
    segment_steps=$(( 64800 * 2 / (3 * slower_seconds) ))
    (( segment_steps < 1 )) && segment_steps=1
    (( segment_steps > 20 )) && segment_steps=20

    profile_summary=${PROFILE_ROOT}/profile_reports/${attempt_name}_summary.json
    python - "${profile_summary}" "${micro_batch}" "${phase1_seconds}" \
      "${phase2_seconds}" "${segment_steps}" "${max_checkpoints}" <<'PY'
import json
import sys
from pathlib import Path

output, micro, step1, step2, segment, keep = sys.argv[1:]
payload = {
    "ok": True,
    "micro_batch": int(micro),
    "step_1_seconds": int(step1),
    "step_2_seconds": int(step2),
    "slower_step_seconds": max(int(step1), int(step2)),
    "safety_multiplier": 1.5,
    "training_budget_seconds": 64800,
    "segment_steps": int(segment),
    "max_checkpoints": int(keep),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

    # The profile has already proved model/optimizer/scheduler/RNG restoration.
    # Retain its controller states, traces and validation reports, but release
    # the large profile-only actor shards before the formal run starts.
    python - "${PROFILE_ROOT}/checkpoints" "${checkpoint_dir}" \
      "${PROFILE_ROOT}/profile_reports/${attempt_name}_checkpoint_thinning.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

parent = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
report_path = Path(sys.argv[3])
if root.parent != parent or not root.name.startswith("profile_mb"):
    raise SystemExit(f"refusing unsafe profile checkpoint thinning: {root}")
removed = []
released = 0
for actor in sorted(root.glob("global_step_*/actor")):
    if actor.parent.parent != root or not actor.parent.name.startswith("global_step_"):
        raise SystemExit(f"refusing unsafe actor checkpoint deletion: {actor}")
    size = sum(path.stat().st_size for path in actor.rglob("*") if path.is_file())
    shutil.rmtree(actor)
    removed.append(str(actor))
    released += size
report_path.write_text(json.dumps({
    "reason": "profile restoration already validated; preserve formal checkpoint capacity",
    "removed_actor_directories": removed,
    "released_bytes": released,
    "retained": ["bace_collector_state.json", "data.pt", "trace and validation reports"],
}, indent=2, sort_keys=True) + "\n")
PY

    # A two-process recovery profile cannot expose host-thread accumulation
    # across multiple steps in one Python/Ray process.  Gate the formal chain
    # on a three-step stability job before it is submitted.
    sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
        --export="ALL,STABILITY_MICRO_BATCH=${micro_batch},STABILITY_SEGMENT_STEPS=${segment_steps},STABILITY_MAX_CHECKPOINTS=${max_checkpoints}" \
        "${SCRIPT_DIR}/slurm_alfworld_h100_4gpu_stability.sbatch"
    exit 0
done

echo "All configured micro-batches failed with confirmed CUDA OOM." >&2
exit 1
