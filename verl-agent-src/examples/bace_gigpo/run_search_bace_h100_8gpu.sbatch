#!/usr/bin/env bash
# Formal 200-update BACE Search-R1 run, aligned with the corrected 3B GiGPO
# reference in AFH/examples/gigpo_trainer/run_search_gigpo_h100_8gpu.sbatch.
#
# Two nodes are required because each node hosts one full-index retriever,
# sharded over its four H100s.  The training job uses all eight H100s through
# one two-node Ray cluster.  Re-submitting this same script resumes the latest
# complete checkpoint, including dataloader and BACE family-history state.
#SBATCH --job-name=bace_search_c0
#SBATCH --partition=c23g
#SBATCH --account=rwth2089
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=23:30:00
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

SEED=${BACE_SEARCH_SEED:-0}
TOTAL_TRAINING_STEPS=200
SAVE_FREQ=50
MAX_CHECKPOINTS=2
RUN_ID=${BACE_SEARCH_RUN_ID:-bace_search_c0_c045_erv0025_qwen2.5_3b_seed${SEED}_200step_v1}
RUN_ROOT=${EXP_ROOT}/runs/${RUN_ID}
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_ID}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_ID}
ARTIFACT_DIR=${RUN_ROOT}/bace_artifacts
ROLLOUT_DIR=${RUN_ROOT}/rollout_trajectories
METADATA_DIR=${RUN_ROOT}/metadata
REPORT_DIR=${RUN_ROOT}/trace_validation
INVOCATION_DIR=${RUN_ROOT}/segments/${SLURM_JOB_ID:-manual}
PORT=${SEARCH_RETRIEVER_PORT:-18000}
PYTHON_BIN=${TRAIN_ENV}/bin/python
MANIFEST_TOOL=${WORKSPACE_ROOT}/AFH/examples/gigpo_trainer/reference_manifest.py
PRUNE_TOOL=${REPO_ROOT}/examples/bace_gigpo/prune_checkpoints.py

[[ ${RUN_ID} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe run id: ${RUN_ID}" >&2; exit 2; }

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TENSORBOARD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Keep vLLM/torch-compile products away from the small home quota.
export XDG_CACHE_HOME=${EXP_ROOT}/runtime_cache/xdg
export TORCHINDUCTOR_CACHE_DIR=${EXP_ROOT}/runtime_cache/torchinductor

for required in "${REPO_ROOT}" "${TRAIN_ENV}" "${RETRIEVER_ENV}" "${MODEL_PATH}" \
    "${SEARCH_ROOT}/e5_Flat.index" "${SEARCH_ROOT}/wiki-18.jsonl" \
    "${SEARCH_ROOT}/e5-base-v2" "${TRAIN_FILE}" "${VAL_FILE}" \
    "${MANIFEST_TOOL}" "${PRUNE_TOOL}"; do
    [[ -e ${required} ]] || { echo "Missing required asset: ${required}" >&2; exit 2; }
done
[[ -f ${MODEL_PATH}/model.safetensors.index.json || -f ${MODEL_PATH}/model.safetensors ]] || {
    echo "No safetensors model weights at ${MODEL_PATH}" >&2; exit 2;
}
"${RETRIEVER_ENV}/bin/python" - <<'PY'
import faiss
required = ("GpuMultipleClonerOptions", "StandardGpuResources", "index_cpu_to_all_gpus")
missing = [name for name in required if not hasattr(faiss, name)]
if missing:
    raise SystemExit("FAISS GPU Python bindings unavailable: " + ", ".join(missing))
print("FAISS GPU Python bindings: OK", faiss.__version__)
PY

mkdir -p "${EXP_ROOT}/slurm" "${INVOCATION_DIR}" "${METADATA_DIR}" \
    "${REPORT_DIR}" "${ARTIFACT_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}" \
    "${TENSORBOARD_DIR}" "${EXP_ROOT}/retriever_cache/formal_slot_0/hf_home" \
    "${EXP_ROOT}/retriever_cache/formal_slot_0/datasets" \
    "${EXP_ROOT}/retriever_cache/formal_slot_1/hf_home" \
    "${EXP_ROOT}/retriever_cache/formal_slot_1/datasets" \
    "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"

available_kib=$(df -Pk "${EXP_ROOT}" | awk 'NR==2 {print $4}')
available_inodes=$(df -Pi "${EXP_ROOT}" | awk 'NR==2 {print $4}')
(( available_kib >= 160 * 1024 * 1024 )) || {
    echo "Less than 160 GiB free for two rolling checkpoints and two retriever caches." >&2; exit 3;
}
(( available_inodes >= 100000 )) || { echo "Less than 100,000 inodes free." >&2; exit 3; }

if [[ -f ${RUN_ROOT}/summary.json ]]; then
    status=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${RUN_ROOT}/summary.json")
    [[ ${status} != complete ]] || { echo "Run ${RUN_ID} is already complete; refusing to overwrite it." >&2; exit 4; }
fi

# Recognize and safely rotate checkpoints written by a previous Slurm process.
# The trainer also performs this rotation after every newly committed save.
if [[ -f ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]]; then
    "${PYTHON_BIN}" "${PRUNE_TOOL}" "${CHECKPOINT_DIR}" --keep "${MAX_CHECKPOINTS}" \
        --output "${INVOCATION_DIR}/checkpoint_prune_before.json"
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#nodes[@]} == 2 )) || { echo "Expected exactly two nodes, got ${#nodes[@]}." >&2; exit 4; }
ipv4_for_node() {
    srun --nodes=1 --ntasks=1 -w "$1" hostname -I | awk '{for (i = 1; i <= NF; ++i) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) { print $i; exit }}'
}
head_node=${nodes[0]}
head_ip=$(ipv4_for_node "${head_node}")
[[ -n ${head_ip} ]] || { echo "Could not determine Ray head IPv4 address." >&2; exit 4; }
ray_address=${head_ip}:6379

retriever_pids=()
ray_pids=()
profiler_pids=()
cleanup() {
    for pid in "${profiler_pids[@]:-}" "${retriever_pids[@]:-}" "${ray_pids[@]:-}"; do
        [[ -n ${pid} ]] && kill "${pid}" 2>/dev/null || true
    done
    for node in "${nodes[@]}"; do
        srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
            env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ray stop --force >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

endpoints_file=${INVOCATION_DIR}/retriever_endpoints.txt
>"${endpoints_file}"
for node_index in "${!nodes[@]}"; do
    node=${nodes[${node_index}]}
    node_ip=$(ipv4_for_node "${node}")
    [[ -n ${node_ip} ]] || { echo "Could not determine address of ${node}." >&2; exit 4; }
    printf '%s\n' "${node} ${node_ip}:${PORT}" >>"${endpoints_file}"
    retriever_log=${INVOCATION_DIR}/retriever_${node}.log
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc "
        source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
        conda activate '${RETRIEVER_ENV}'
        cd '${REPO_ROOT}'
        export HF_HOME='${EXP_ROOT}/retriever_cache/formal_slot_${node_index}/hf_home'
        export HF_DATASETS_CACHE='${EXP_ROOT}/retriever_cache/formal_slot_${node_index}/datasets'
        exec '${RETRIEVER_ENV}/bin/python' -u examples/search/retriever/retrieval_server.py \\
            --index_path '${SEARCH_ROOT}/e5_Flat.index' \\
            --corpus_path '${SEARCH_ROOT}/wiki-18.jsonl' --topk 3 \\
            --retriever_name e5 --retriever_model '${SEARCH_ROOT}/e5-base-v2' \\
            --faiss_gpu --port '${PORT}'
    " >"${retriever_log}" 2>&1 &
    retriever_pids+=("$!")
done

for node_index in "${!nodes[@]}"; do
    node=${nodes[${node_index}]}
    endpoint=$(awk -v n="${node}" '$1 == n {print $2}' "${endpoints_file}")
    ready=false
    for attempt in {1..360}; do
        if curl --silent --show-error --fail --max-time 20 -X POST "http://${endpoint}/retrieve" \
            -H 'Content-Type: application/json' \
            -d '{"query":"What is the capital of France?","topk":1}' \
            >"${INVOCATION_DIR}/retriever_probe_${node}.json"; then
            ready=true
            break
        fi
        if ! kill -0 "${retriever_pids[${node_index}]}" 2>/dev/null; then
            echo "Retriever on ${node} exited before becoming ready." >&2
            tail -n 120 "${INVOCATION_DIR}/retriever_${node}.log" >&2 || true
            exit 4
        fi
        sleep 5
    done
    ${ready} || {
        echo "Retriever on ${node} was not ready within thirty minutes." >&2
        tail -n 120 "${INVOCATION_DIR}/retriever_${node}.log" >&2 || true
        exit 4
    }
done

# Slurm injects ROCm visibility variables even on NVIDIA nodes.  Remove them
# in the daemon process itself so every Ray actor inherits a CUDA-only view.
srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u VLLM_ATTENTION_BACKEND \
    ray start --head --node-ip-address="${head_ip}" --port=6379 \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block \
    >"${INVOCATION_DIR}/ray_head.log" 2>&1 &
ray_pids+=("$!")
sleep 8
for node in "${nodes[@]:1}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u VLLM_ATTENTION_BACKEND \
        ray start --address="${ray_address}" --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=4 --block >"${INVOCATION_DIR}/ray_${node}.log" 2>&1 &
    ray_pids+=("$!")
done
sleep 8

search_urls=$(awk '{printf "\"http://%s/retrieve\",", $2}' "${endpoints_file}")
search_urls="[${search_urls%,}]"
COMMAND_FILE=${INVOCATION_DIR}/resolved_command.sh

# Every non-BACE training/Search setting below matches the corrected GiGPO
# 3B reference.  total_leaf_budget=5 preserves its per-task rollout count;
# BACE dynamically trades natural roots for selected-worker branches within
# that same fixed budget.
ppo_command=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo ++ray_init.address="${ray_address}"
    algorithm.adv_estimator=bace_gigpo algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.topology=dynamic algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed algorithm.bace.branch_execution_mode=selected_worker
    algorithm.bace.branch_pool_mode=main_reuse algorithm.bace.root_active_executor=true
    algorithm.bace.total_leaf_budget=5 algorithm.bace.min_natural_roots=2
    algorithm.bace.capacity_correction_batch_size=1 algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.45 algorithm.bace.history_initial_mean=0.10
    algorithm.bace.history_initial_strength=2.0 algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1 algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0 algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold=0.0025
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
    data.train_batch_size=256 data.val_batch_size=512
    data.max_prompt_length=4096 data.max_response_length=512
    data.filter_overlong_prompts=True data.truncation=left data.return_raw_chat=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=512 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01
    algorithm.use_kl_in_reward=False algorithm.gamma=0.95
    algorithm.gigpo.step_advantage_w=1.0 algorithm.gigpo.mode=mean_std_norm
    algorithm.gigpo.enable_similarity=True algorithm.gigpo.similarity_thresh=0.9
    env.env_name=search env.seed="${SEED}" env.max_steps=4 env.rollout.n=5 env.history_length=4
    "env.search.search_url=${search_urls}"
    trainer.critic_warmup=0 "trainer.logger=['console','tensorboard']"
    trainer.project_name=verl_agent_search_bace trainer.experiment_name="${RUN_ID}"
    trainer.n_gpus_per_node=4 trainer.nnodes=2
    trainer.save_freq="${SAVE_FREQ}" trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    trainer.max_global_ckpt_to_keep="${MAX_CHECKPOINTS}"
    trainer.default_local_dir="${CHECKPOINT_DIR}" trainer.rollout_data_dir="${ROLLOUT_DIR}"
    trainer.resume_mode=auto trainer.test_freq=50 trainer.total_epochs=1
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.val_before_train=False trainer.validate_on_last_step=True
)

printf '%q ' "${ppo_command[@]}" >"${COMMAND_FILE}"
printf '\n' >>"${COMMAND_FILE}"
"${ppo_command[@]}" --cfg job --resolve >"${INVOCATION_DIR}/resolved_config.yaml"
"${PYTHON_BIN}" "${MANIFEST_TOOL}" --repo "${REPO_ROOT}" \
    --asset "${MODEL_PATH}" --asset "${TRAIN_FILE}" --asset "${VAL_FILE}" \
    --command-file "${COMMAND_FILE}" \
    --source-archive "${RUN_ROOT}/metadata/source_snapshot.tar.gz" \
    --output "${METADATA_DIR}/run_${SLURM_JOB_ID:-manual}.json"
stat --format='%n bytes=%s mtime=%y' "${SEARCH_ROOT}/e5_Flat.index" \
    "${SEARCH_ROOT}/wiki-18.jsonl" "${SEARCH_ROOT}/e5-base-v2/model.safetensors" \
    >"${INVOCATION_DIR}/retrieval_assets.txt"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "Dry-run complete: ${COMMAND_FILE}"
    exit 0
fi

for node in "${nodes[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc \
        "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 10" \
        >"${INVOCATION_DIR}/gpu_${node}.csv" 2>&1 &
    profiler_pids+=("$!")
done

set +e
"${ppo_command[@]}" 2>&1 | tee "${INVOCATION_DIR}/trainer.log"
train_status=${PIPESTATUS[0]}
set -e
printf 'train_exit_code=%s\nend_time=%s\n' "${train_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"${INVOCATION_DIR}/completion.txt"
(( train_status == 0 )) || exit "${train_status}"

grep -Eq "training/global_step['\"]?:[[:space:]]*${TOTAL_TRAINING_STEPS}|training/global_step[^0-9]+${TOTAL_TRAINING_STEPS}" \
    "${INVOCATION_DIR}/trainer.log"
! grep -Eqi 'CUDA out of memory|OutOfMemoryError|RayTaskError|NCCL[^[:space:]]* error' \
    "${INVOCATION_DIR}/trainer.log"
"${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${SLURM_JOB_ID:-manual}_final.json"
"${PYTHON_BIN}" "${PRUNE_TOOL}" "${CHECKPOINT_DIR}" --keep "${MAX_CHECKPOINTS}" \
    --output "${INVOCATION_DIR}/checkpoint_prune_after.json"

"${PYTHON_BIN}" - "${ARTIFACT_DIR}/step_$(printf '%08d' "${TOTAL_TRAINING_STEPS}")" \
    "${CHECKPOINT_DIR}" "${TOTAL_TRAINING_STEPS}" <<'PY'
import json, sys
from pathlib import Path
step, checkpoints, total = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
summary = json.loads((step / "summary.json").read_text())
manifest = json.loads((step / "manifest.json").read_text())
trace = summary["bace_metrics"]
assert summary["status"] == "complete"
assert manifest["variant"] == "batch_erv_exact"
assert manifest["staged_root_batching"] == "packed"
assert manifest["branch_execution_mode"] == "selected_worker"
assert manifest["anchor_similarity_enabled"] is True
assert float(manifest["anchor_similarity_threshold"]) == 0.9
assert manifest["allow_initial_search_anchor"] is True
assert float(manifest["competence_threshold"]) == 0.45
assert float(manifest["batch_erv_threshold"]) == 0.0025
assert float(trace["batch_erv_exact"]) == 1.0
assert float(trace["staged_root_batching_packed"]) == 1.0
assert (step / "roots.jsonl").stat().st_size > 0
assert (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip() == str(total)
final = checkpoints / f"global_step_{total}"
assert (final / "actor").is_dir()
assert (final / "data.pt").is_file()
state = json.loads((final / "bace_collector_state.json").read_text())
assert int(state["global_step"]) == total and isinstance(state["collector"], dict)
print("Search BACE formal final gate passed")
PY

printf '{"status":"complete","run_id":"%s","steps":%s,"save_freq":%s,"max_checkpoints":%s,"competence_threshold":0.45,"batch_erv_threshold":0.0025}\n' \
    "${RUN_ID}" "${TOTAL_TRAINING_STEPS}" "${SAVE_FREQ}" "${MAX_CHECKPOINTS}" \
    >"${RUN_ROOT}/summary.json"
echo "Search BACE formal run completed: ${RUN_ID}"
