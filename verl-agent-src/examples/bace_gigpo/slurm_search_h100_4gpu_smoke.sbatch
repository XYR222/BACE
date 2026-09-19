#!/usr/bin/env bash
# One-step Search/HotpotQA BACE smoke.  This job never submits a formal run.
# Four H100s are required because the 60 GiB FAISS index is sharded over every
# visible GPU before Ray/PPO starts, matching the proven Search GiGPO layout.
#SBATCH --job-name=bace_search_smoke_4g
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/search-qwen2.5-3b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/search-qwen2.5-3b-exact/slurm/%x-%j.err
set -euo pipefail

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
TRAIN_ENV=${WORKSPACE_ROOT}/verl-agent
RETRIEVER_ENV=/hpcwork/rwth2089/xsz96350/work-BACE/retriever
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-3B-Instruct
SEARCH_ROOT=/hpcwork/rwth2089/xsz96350/work-BACE/searchR1
TRAIN_FILE=/hpcwork/rwth2089/xsz96350/work-BACE/searchR1_processed_direct/train.parquet
VAL_FILE=/hpcwork/rwth2089/xsz96350/work-BACE/searchR1_processed_direct/test.parquet
EXP_ROOT=/hpcwork/rwth2089/xsz96350/work-BACE/experiments/search-qwen2.5-3b-exact
RUN_ID=bace_search_exact_smoke4g_${SLURM_JOB_ID:-manual}
RUN_ROOT=${EXP_ROOT}/validation/${RUN_ID}
ARTIFACT_DIR=${RUN_ROOT}/bace_artifacts
ROLLOUT_DIR=${RUN_ROOT}/rollout_trajectories
METADATA_DIR=${RUN_ROOT}/metadata
REPORT_DIR=${RUN_ROOT}/trace_validation
TENSORBOARD_DIR=${RUN_ROOT}/tensorboard
LOG_FILE=${RUN_ROOT}/training.log
PORT=${SEARCH_SMOKE_PORT:-18000}
PYTHON_BIN=${TRAIN_ENV}/bin/python

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" TENSORBOARD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[[ ${SLURM_GPUS_ON_NODE:-0} -eq 4 ]] || { echo 'Expected exactly four H100s.' >&2; exit 2; }
for required in "${REPO_ROOT}" "${TRAIN_ENV}" "${RETRIEVER_ENV}" "${MODEL_PATH}" \
    "${SEARCH_ROOT}/e5_Flat.index" "${SEARCH_ROOT}/wiki-18.jsonl" \
    "${SEARCH_ROOT}/e5-base-v2" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e ${required} ]] || { echo "Missing required asset: ${required}" >&2; exit 2; }
done
[[ -f ${MODEL_PATH}/model.safetensors.index.json || -f ${MODEL_PATH}/model.safetensors ]] || {
    echo 'No model safetensors weights found.' >&2; exit 2;
}

mkdir -p "${ARTIFACT_DIR}" "${ROLLOUT_DIR}" "${METADATA_DIR}" \
    "${REPORT_DIR}" "${TENSORBOARD_DIR}" "${EXP_ROOT}/slurm" \
    "${EXP_ROOT}/retriever_cache/hf_home" "${EXP_ROOT}/retriever_cache/datasets"
# The corpus-derived Arrow cache is reusable and stays on the 3 TiB project.
export HF_HOME=${EXP_ROOT}/retriever_cache/hf_home
export HF_DATASETS_CACHE=${EXP_ROOT}/retriever_cache/datasets

RETRIEVER_LOG=${RUN_ROOT}/retriever.log
cleanup() {
    [[ -n ${retriever_pid:-} ]] && kill "${retriever_pid}" 2>/dev/null || true
    [[ -n ${monitor_pid:-} ]] && kill "${monitor_pid}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

"${RETRIEVER_ENV}/bin/python" - <<'PY' > "${METADATA_DIR}/faiss_preflight.txt"
import faiss
required = ("GpuMultipleClonerOptions", "StandardGpuResources", "index_cpu_to_all_gpus")
missing = [name for name in required if not hasattr(faiss, name)]
if missing:
    raise SystemExit("FAISS GPU bindings unavailable: " + ", ".join(missing))
print("FAISS GPU bindings: OK", faiss.__version__)
PY

(
    source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
    conda activate "${RETRIEVER_ENV}"
    cd "${REPO_ROOT}"
    exec "${RETRIEVER_ENV}/bin/python" -u examples/search/retriever/retrieval_server.py \
        --index_path "${SEARCH_ROOT}/e5_Flat.index" \
        --corpus_path "${SEARCH_ROOT}/wiki-18.jsonl" --topk 3 \
        --retriever_name e5 --retriever_model "${SEARCH_ROOT}/e5-base-v2" \
        --faiss_gpu --port "${PORT}"
) > "${RETRIEVER_LOG}" 2>&1 &
retriever_pid=$!
(
    while kill -0 "${retriever_pid}" 2>/dev/null; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
        sleep 10
    done
) > "${METADATA_DIR}/gpu_monitor.log" 2>&1 &
monitor_pid=$!

for attempt in {1..360}; do
    if curl --silent --show-error --fail --max-time 20 \
        -X POST "http://127.0.0.1:${PORT}/retrieve" \
        -H 'Content-Type: application/json' \
        -d '{"query":"What is the capital of France?","topk":1}' \
        > "${METADATA_DIR}/retriever_probe.json"; then
        break
    fi
    kill -0 "${retriever_pid}" 2>/dev/null || {
        tail -n 120 "${RETRIEVER_LOG}" >&2; exit 3;
    }
    sleep 5
done
[[ -s ${METADATA_DIR}/retriever_probe.json ]] || {
    echo 'Retriever was not ready within thirty minutes.' >&2; exit 3;
}

# CPU/unit checks plus a real two-worker restore/transition probe happen before PPO.
"${PYTHON_BIN}" -m pytest -q tests/bace_gigpo/test_search_replay.py \
    > "${METADATA_DIR}/search_replay_tests.txt"
"${PYTHON_BIN}" examples/bace_gigpo/validate_search_selected_worker_real_env.py \
    --search-url "http://127.0.0.1:${PORT}/retrieve" \
    --output "${REPORT_DIR}/selected_worker_real_env.json"

ray stop --force 2>/dev/null || true
ray start --head --port=6379 --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --num-gpus=4 \
    > "${METADATA_DIR}/ray.log" 2>&1

ppo_command=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo ++ray_init.address=auto
    algorithm.adv_estimator=bace_gigpo algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.topology=dynamic algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed algorithm.bace.branch_execution_mode=selected_worker
    algorithm.bace.branch_pool_mode=main_reuse algorithm.bace.root_active_executor=true
    algorithm.bace.total_leaf_budget=5 algorithm.bace.min_natural_roots=2
    algorithm.bace.capacity_correction_batch_size=1 algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.5 algorithm.bace.history_initial_mean=0.10
    algorithm.bace.history_initial_strength=2.0 algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1 algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0 algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold=0.005
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity
    algorithm.bace.tie_break_identity_mode=stable_v1
    algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.anchor_similarity_enabled=true
    algorithm.bace.anchor_similarity_threshold=0.9
    algorithm.bace.allow_initial_search_anchor=true
    algorithm.bace.credit_mode=current algorithm.bace.tree_credit_mode=current
    algorithm.bace.macro_normalization_mode=stable_occurrence
    algorithm.bace.tree_ppo_padding_mode=copy_trainable
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
    algorithm.bace.artifacts.include_token_arrays=true algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true algorithm.bace.replay.max_origin_retries=1
    data.train_files="${TRAIN_FILE}" data.val_files="${VAL_FILE}"
    data.train_batch_size=8 data.val_batch_size=8
    data.max_prompt_length=4096 data.max_response_length=512
    data.filter_overlong_prompts=true data.truncation=left data.return_raw_chat=true
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.actor.ppo_mini_batch_size=512
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_chunked_prefill=false
    actor_rollout_ref.rollout.enforce_eager=false actor_rollout_ref.rollout.free_cache_engine=false
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.use_invalid_action_penalty=true
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01
    algorithm.use_kl_in_reward=false algorithm.gamma=0.95
    algorithm.gigpo.step_advantage_w=1.0 algorithm.gigpo.mode=mean_std_norm
    algorithm.gigpo.enable_similarity=true algorithm.gigpo.similarity_thresh=0.9
    env.env_name=search env.seed=0 env.max_steps=4 env.rollout.n=5 env.history_length=4
    env.search.search_url="http://127.0.0.1:${PORT}/retrieve"
    env.search.topk=3 env.search.timeout=60 env.search.log_requests=false
    trainer.critic_warmup=0 "trainer.logger=['console','tensorboard']"
    trainer.project_name=verl_agent_search_bace trainer.experiment_name="${RUN_ID}"
    trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.save_freq=-1
    trainer.default_local_dir="${RUN_ROOT}/checkpoints_disabled"
    trainer.rollout_data_dir="${ROLLOUT_DIR}" trainer.resume_mode=disable
    trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=1
    trainer.val_before_train=false trainer.validate_on_last_step=false
)

printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
"${ppo_command[@]}" --cfg job --resolve > "${METADATA_DIR}/resolved_config.yaml"
( find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
    -print0 | sort -z | xargs -0 sha256sum ) > "${METADATA_DIR}/source_manifest.sha256"
"${PYTHON_BIN}" - <<'PY' > "${METADATA_DIR}/preflight.json"
import json, os, torch
report = {
    "ok": torch.cuda.is_available() and torch.cuda.device_count() == 4,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["ok"]:
    raise SystemExit(4)
PY
nvidia-smi -q > "${METADATA_DIR}/nvidia_smi_q.txt"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
"${PYTHON_BIN}" -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"

set +e
"${ppo_command[@]}" 2>&1 | tee "${LOG_FILE}"
train_status=${PIPESTATUS[0]}
set -e
printf 'train_exit_code=%s\n' "${train_status}" > "${METADATA_DIR}/completion.txt"
(( train_status == 0 )) || exit "${train_status}"
grep -Eq "training/global_step['\"]?:[[:space:]]*1|training/global_step[^0-9]+1" "${LOG_FILE}"
! grep -Eqi 'CUDA out of memory|OutOfMemoryError|RayTaskError|NCCL[^[:space:]]* error' "${LOG_FILE}"
"${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${SLURM_JOB_ID:-local}_final.json"

"${PYTHON_BIN}" - "${ARTIFACT_DIR}/step_00000001" "${REPORT_DIR}/selected_worker_real_env.json" <<'PY'
import json, sys
from pathlib import Path
step, replay_report = Path(sys.argv[1]), Path(sys.argv[2])
summary = json.loads((step / "summary.json").read_text())
manifest = json.loads((step / "manifest.json").read_text())
probe = json.loads(replay_report.read_text())
metrics = summary["bace_metrics"]
assert summary["status"] == "complete" and probe["ok"] is True
assert manifest["variant"] == "batch_erv_exact"
assert manifest["staged_root_batching"] == "packed"
assert manifest["branch_execution_mode"] == "selected_worker"
assert manifest["anchor_similarity_enabled"] is True
assert float(manifest["anchor_similarity_threshold"]) == 0.9
assert float(metrics["batch_erv_exact"]) == 1.0
assert float(metrics["staged_root_batching_packed"]) == 1.0
assert (step / "roots.jsonl").stat().st_size > 0
print("Search BACE one-step smoke passed")
PY

printf '{"status":"complete","run_id":"%s","steps":1,"checkpoint_saved":false,"formal_job_submitted":false}\n' \
    "${RUN_ID}" > "${RUN_ROOT}/summary.json"
echo "Search BACE smoke completed; no formal job was submitted."
